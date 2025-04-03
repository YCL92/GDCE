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
from util.data_util import CustomSet
from util.model_util import Classifier, trainClassifier, valClassifier


def main(subset, ckpt_dir, clamp):
    # common params
    initRandomSeed()
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    img_folder = "images-%s" % subset

    # load config file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    if "embed" in config["dataset_dir"].lower():
        dataset_name = "embed"
        pred_tag = "Selenia"
        num_class = 4
    else:
        dataset_name = "rsna"
        pred_tag = "Brand_B"
        num_class = 2

    if ckpt_dir is not None:
        log_name = strftime(dataset_name + f"-patho-{clamp}-%Y%m%d_%H%M%S")

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

    for fold in range(1, 5 + 1):
        logging.info(f"\n\nStart training @fold{fold:d}...")

        # set up dataloaders
        train_dataset = CustomSet(
            config["dataset_dir"],
            img_folder,
            f"./data/{dataset_name}-5fold-csv/train-fold{fold:d}.csv",
            transform=train_transform,
            pred_tag=pred_tag,
            clamp=clamp,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_worker"]
        )
        val_dataset = CustomSet(
            config["dataset_dir"],
            img_folder,
            f"./data/{dataset_name}-5fold-csv/val-fold{fold:d}.csv",
            transform=val_transform,
            pred_tag=pred_tag,
            clamp=clamp,
        )
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=config["num_worker"])

        # compile model
        if ckpt_dir is not None:
            checkpoint_path = os.path.join("./checkpoint", ckpt_dir, "checkpoint.pt")
        else:
            checkpoint_path = None
        classifier = Classifier(backbone=config["model_name"], num_class=num_class, checkpoint_path=checkpoint_path).to(
            device
        )

        optim = Adam(classifier.parameters(), lr=config["learning_rate"])
        sched = ReduceLROnPlateau(optim, mode="max", factor=0.1, patience=config["patience"], threshold_mode="abs")
        monitor = ModelSaver(mode="max", save_dir=save_dir, file_name=f"checkpoint-fold{fold}.pt")

        # training
        for epoch in range(config["max_epoch"]):
            train_loss = trainClassifier(train_loader, classifier, optim, device=device)
            val_loss, val_worst = valClassifier(val_loader, classifier, device=device)
            sched.step(val_worst)
            monitor.step(val_worst, classifier, epoch + 1)

            # print results
            out_msg = f"Epoch [{epoch + 1}/{config['max_epoch']}], Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val worst: {val_worst:.4f}"
            logging.info(out_msg)
            print(out_msg)

            # early-stopping
            current_lr = optim.param_groups[0]["lr"]
            if current_lr < config["min_lr_ratio"] * config["learning_rate"]:
                print("Early-stopping has reached, terminate training.")
                break


if __name__ == "__main__":
    subset = "full_512"
    clamp = "full"
    ckpt_dir = "resnet-pretrain-radimagenet_vanilla-20241212_160353"

    main(subset, ckpt_dir, clamp)
