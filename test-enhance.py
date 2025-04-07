import os

import numpy as np
import torch as t
import yaml
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from util.data_util import CustomSet
from util.model_util import Classifier, Enhancer, testEnhancer


def main(subset, machine_config):
    # common params
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    img_folder = "images-%s" % subset
    source_machine, enhancer_dir, classifier_dir = machine_config

    # load config file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    if source_machine in ["Clearview", "Senograph"]:
        dataset_name = "embed"
        dataset_dir = config["embed_dir"]
        task_type = "multi-class"
        num_class = 4
    else:
        dataset_name = "rsna"
        dataset_dir = config["rsna_dir"]
        task_type = "binary"
        num_class = 2

    # augmentation ops
    val_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(t.float32),
            v2.Lambda(lambda x: x / 65535),
            v2.Resize((config["img_size"], config["img_size"])),
        ]
    )

    test_dataset = CustomSet(
        dataset_dir,
        img_folder,
        f"./data/{dataset_name}-5fold-csv/test.csv",
        transform=val_transform,
        pred_tag=source_machine,
        clamp="full",
    )
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], num_workers=config["num_worker"])

    result_list = []

    for fold in range(1, 6):
        # compile models
        checkpoint_path = os.path.join("./checkpoint", enhancer_dir, f"checkpoint-fold{fold}.pt")
        enhancer = Enhancer(
            num_layer=config["model_n_layer"], num_iter=config["model_n_iter"], checkpoint_path=checkpoint_path
        ).to(device)

        checkpoint_path = os.path.join("./checkpoint", classifier_dir, f"checkpoint-fold{fold}.pt")
        classifier = Classifier(backbone=config["model_name"], num_class=num_class, checkpoint_path=checkpoint_path).to(
            device
        )

        # testing
        if task_type == "multi-class":
            test_auc, test_confmat = testEnhancer(test_loader, enhancer, classifier, task=task_type, device=device)
            result_list.append({"auc": test_auc, "confmat": test_confmat})
        else:
            test_prec, test_recall = testEnhancer(test_loader, enhancer, classifier, task=task_type, device=device)
            result_list.append({"precision": test_prec, "recall": test_recall})

    # summarize
    print("\n===== Average Results over 5 Folds =====")
    if task_type == "multi-class":
        aucs = [r["auc"] for r in result_list]
        confmats = [r["confmat"] for r in result_list]
        mean_auc = np.nanmean(aucs)
        mean_confmat = np.mean(confmats, axis=0)

        print(f"Mean AUC: {mean_auc:.4f}")
        print("Mean Confusion Matrix (row-normalized):")
        print(mean_confmat)

    else:
        precisions = [r["precision"] for r in result_list]
        recalls = [r["recall"] for r in result_list]
        mean_prec = np.mean(precisions)
        mean_recall = np.mean(recalls)

        print(f"Mean Precision: {mean_prec:.4f}")
        print(f"Mean Recall: {mean_recall:.4f}")


if __name__ == "__main__":
    subset = "full_512"
    machine_config = ["Clearview", "embed-clearview-20250406_142325", "embed-patho-full-20250317_143344"]
    # machine_config = ["Senograph", "embed-senograph-20250406_114917", "embed-patho-full-20250317_143344"]
    # machine_config = ['Brand A', "rsna-brand_a-20250406_153317", "rsna-patho-full-20250317_234039"]

    main(subset, machine_config)
