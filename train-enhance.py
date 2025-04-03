import logging
import os
from time import strftime

import torch as t
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from util.commom import ModelSaver, initRandomSeed
from util.data_util import BatchSampler, CustomSet
from util.model_util import Classifier, Enhancer, VGGLoss, trainEnhancerVGG, valEnhancer


def main(subset, machine_config, vgg_dir):
    # common params
    initRandomSeed()
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    img_folder = "images-%s" % subset
    source_machine, classifier_dir = machine_config

    # load config file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    if "embed" in config["dataset_dir"].lower():
        dataset_name = "embed"
        target_machine = "Selenia"
        num_class = 4
        loss_lambda = 0.9
    else:
        dataset_name = "rsna"
        target_machine = "Brand_B"
        num_class = 2
        loss_lambda = 0.1

    log_name = strftime(dataset_name + f"-{source_machine.lower()}-%Y%m%d_%H%M%S")

    # config logging
    save_dir = os.path.join("./checkpoint", log_name)
    os.makedirs(save_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(save_dir, "training.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # augmentation ops
    train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(t.float32),
            v2.Lambda(lambda x: x / 65535),
            v2.Resize((config["img_size"], config["img_size"])),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
        ]
    )
    val_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(t.float32),
            v2.Lambda(lambda x: x / 65535),
            v2.Resize((config["img_size"], config["img_size"])),
        ]
    )

    for fold in range(1, 6):
        logging.info("\n\nStart training @fold%d..." % fold)

        # set up dataloaders
        train_dataset_src = CustomSet(
            config["dataset_dir"],
            img_folder,
            f"./data/{dataset_name}-5fold-csv/train-fold1.csv",
            transform=train_transform,
            pred_tag=source_machine,
            clamp="full",
        )
        sampler_src = BatchSampler(train_dataset_src, 3, config["batch_size"])
        train_loader_src = DataLoader(train_dataset_src, num_workers=config["num_worker"], batch_sampler=sampler_src)

        train_dataset_tgt = CustomSet(
            config["dataset_dir"],
            img_folder,
            f"./data/{dataset_name}-5fold-csv/train-fold1.csv",
            transform=train_transform,
            pred_tag=target_machine,
            clamp="full",
        )
        sampler_tgt = BatchSampler(train_dataset_tgt, 3, config["batch_size"])
        train_loader_tgt = DataLoader(train_dataset_tgt, num_workers=config["num_worker"], batch_sampler=sampler_tgt)
        val_dataset = CustomSet(
            config["dataset_dir"],
            img_folder,
            f"./data/{dataset_name}-5fold-csv/val-fold1.csv",
            transform=val_transform,
            pred_tag=source_machine,
            clamp="full",
        )
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=config["num_worker"])
        test_dataset = CustomSet(
            config["dataset_dir"],
            img_folder,
            f"./data/{dataset_name}-5fold-csv/test.csv",
            transform=val_transform,
            pred_tag=source_machine,
            clamp="full",
        )

        # compile model
        enhancer = Enhancer(num_layer=num_layer, num_iter=num_iter).to(device)
        checkpoint_path = os.path.join("./checkpoint", classifier_dir, f"checkpoint-fold1.pt")
        classifier = Classifier(backbone=config["model_name"], num_class=num_class, checkpoint_path=checkpoint_path).to(
            device
        )
        checkpoint_path = os.path.join("./checkpoint", vgg_dir, f"checkpoint.pt")
        vgg = VGGLoss(checkpoint_path=checkpoint_path, device=device)

        optim = Adam(enhancer.parameters(), lr=config["learning_rate"])
        sched = ReduceLROnPlateau(optim, mode="max", factor=0.1, patience=config["patience"], threshold_mode="abs")
        monitor = ModelSaver(mode="max", save_dir=save_dir, file_name=f"checkpoint-fold{fold}.pt")

        # training
        for epoch in range(config["max_epoch"]):
            train_loss_vgg, train_loss_ce = trainEnhancerVGG(
                [train_loader_src, train_loader_tgt],
                enhancer,
                classifier,
                vgg,
                optim,
                loss_lambda=loss_lambda,
                device=device,
            )
            val_loss, val_worst = valEnhancer(val_loader, enhancer, classifier, device=device)
            sched.step(val_worst)
            monitor.step(val_worst, enhancer, epoch + 1)

            # print results
            out_msg = f"Epoch: {epoch + 1}, Train VGG loss: {train_loss_vgg:.3f}, Train CE loss: {train_loss_ce:.3f}, Val loss: {val_loss:.3f}, Val worst: {val_worst:.3f}"
            logging.info(out_msg)
            print(out_msg)

            # early-stopping
            current_lr = optim.param_groups[0]["lr"]
            if current_lr < config["min_lr_ratio"] * config["learning_rate"]:
                print("Early-stopping has reached, terminate training.")
                break


if __name__ == "__main__":
    subset = "full_512"
    machine_config = ["Brand_A", "rsna-patho-full-20250317_234039"]
    # machine_config = ["Clearview", "embed-patho-full-20250317_143344"]
    # machine_config = ["Senograph", "embed-patho-full-20250317_143344"]
    vgg_dir = "vgg16-pretrain-radimagenet-20250124_070707"

    main(subset, machine_config, vgg_dir)
