import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as t
import yaml
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from util.data_util import CustomSet
from util.model_util import Classifier, Enhancer, testCombined


def main(subset, enhancer_dir, classifier_dir, fold, config, machine):
    # common params
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    img_folder = "images-%s" % subset

    if 'embed' in config['dataset_dir'].lower():
        dataset_name = 'embed'
        num_class = 4
    else:
        dataset_name = 'rsna'
        num_class = 2

    # make folder for result if necessary
    save_dir = "./result"
    os.makedirs(save_dir, exist_ok=True)

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
        config["dataset_dir"],
        img_folder,
        f"./data/{dataset_name}-5fold-csv/test.csv",
        transform=val_transform,
        pred_tag=machine,
        clamp="full",
    )
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], num_workers=config["num_worker"])

    # compile model
    checkpoint_path = os.path.join("./checkpoint", enhancer_dir, f"checkpoint-fold{fold}.pt")
    enhancer = Enhancer(checkpoint_path=checkpoint_path).to(device)
    checkpoint_path = os.path.join("./checkpoint", classifier_dir, f"checkpoint-fold{fold}.pt")
    classifier = Classifier(backbone=config["model_name"], num_class=num_class, checkpoint_path=checkpoint_path).to(
        device)

    # testing
    test_auc, test_confmat = testCombined(test_loader, enhancer, classifier, device=device)

    return test_auc, test_confmat


if __name__ == "__main__":
    asses_list = ["1", "2", "3", "4"]
    # asses_list = ["False", "True"]
    machine_list = ["Clearview", "Selenia", "Senograph"]
    # machine_list = ['A', 'B', 'C']
    subset = "full_512"
    enhancer_dir_dict = {"joint": "embed-clearview-20250308_010625"}
    classifier_dir_dict = {"joint": "embed-patho-full-20250307_164235"}

    # load config file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # make folder for results
    save_dir = os.path.join("./result/patho-confmat")
    os.makedirs(save_dir, exist_ok=True)

    for verbose, enhancer_dir in enhancer_dir_dict.items():
        auc_dict = {machine: [] for machine in machine_list}
        confmat_dict = {machine: [] for machine in machine_list}
        classifier_dir = classifier_dir_dict[verbose]

        for m_idx, machine in enumerate(machine_list):
            for fold in range(1, 6):
                test_auc, test_confmat = main(subset, enhancer_dir, classifier_dir, fold, config, machine)
                auc_dict[machine].append(test_auc)
                confmat_dict[machine].append(test_confmat)

        # Plot the obtained confusion matrix
        fig, axes = plt.subplots(1, len(machine_list), figsize=(3 * len(machine_list), 3), constrained_layout=True)
        fig.suptitle(f"{verbose}")

        for ax, machine in zip(axes, machine_list):
            avg_auc = np.mean(auc_dict[machine])
            avg_confmat = np.mean(confmat_dict[machine], axis=0)

            # Convert to percentages
            row_sums = avg_confmat.sum(axis=1, keepdims=True)
            avg_confmat_percentage = (avg_confmat / row_sums) * 100  # Convert to percentage

            sns.heatmap(
                avg_confmat_percentage,
                annot=True,
                fmt=".1f",  # One decimal place for percentages
                cmap="Blues",
                cbar=False,
                ax=ax,
                xticklabels=asses_list,
                yticklabels=asses_list,
            )
            ax.set_title(f"{machine}-{auc_dict[machine][0]:.2f}")
            ax.set_ylabel("True label (%)")  # Indicate percentage in the label
            ax.set_xlabel("Predicted label")

        fig.subplots_adjust(top=0.85, wspace=0.4)
        # plt.savefig(os.path.join(save_dir, f"combined-{verbose}.png"))
        plt.show()
