import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as t
import yaml
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from util.data_util import CustomSet
from util.model_util import Classifier, testClassifier


def main(subset, ckpt_dir, fold, config, pred_tag='patho', clamp_method="full"):
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
    test_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(t.float32),
            v2.Lambda(lambda x: x / 65535),
            v2.Resize((config["img_size"], config["img_size"]))
        ]
    )
    test_dataset = CustomSet(
        config["dataset_dir"],
        img_folder,
        f"data/{dataset_name}-5fold-csv/test.csv",
        transform=test_transform,
        pred_tag=pred_tag,
        clamp=clamp_method,
    )
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], num_workers=config["num_worker"])

    # compile model
    checkpoint_path = os.path.join("./checkpoint", ckpt_dir, f"checkpoint-fold{fold}.pt")
    model = Classifier(backbone=config["model_name"], num_class=num_class, checkpoint_path=checkpoint_path).to(device)

    # testing
    if dataset_name == 'embed':
        test_auc, test_confmat = testClassifier(test_loader, model, task='multi-class', device=device)

        return test_auc, test_confmat

    else:
        test_prec, test_recall = testClassifier(test_loader, model, task='binary', device=device)

        return test_prec, test_recall


if __name__ == "__main__":
    subset = "full_512"
    clamp_method = "full"
    classifier_dir = 'embed-patho-full-20250317_143344'
    # classifier_dir = "rsna-patho-full-20250317_234039"

    # load config file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # make folder for results
    save_dir = os.path.join("./result/patho-confmat")
    os.makedirs(save_dir, exist_ok=True)

    # for EMBED dataset
    if 'embed' in config['dataset_dir'].lower():
        machine_list = ["Clearview", "Selenia", "Senograph"]
        auc_dict = {machine: [] for machine in machine_list}
        confmat_dict = {machine: [] for machine in machine_list}

        for machine in machine_list:
            for fold in range(1, 6):
                test_auc, test_confmat = main(subset, classifier_dir, fold, config, pred_tag=machine,
                                              clamp_method=clamp_method)
                auc_dict[machine].append(test_auc)
                confmat_dict[machine].append(test_confmat)

        # plot the obtained confusion matrix
        fig, axes = plt.subplots(1, len(machine_list), figsize=(3 * len(machine_list), 3), constrained_layout=True)

        asses_list = ["1", "2", "3", "4"]
        for ax, machine in zip(axes, machine_list):
            avg_auc = np.mean(auc_dict[machine])
            avg_confmat = np.mean(confmat_dict[machine], axis=0)

            # convert to percentages
            row_sums = avg_confmat.sum(axis=1, keepdims=True)
            avg_confmat_percentage = (avg_confmat / row_sums) * 100

            sns.heatmap(
                avg_confmat_percentage,
                annot=True,
                fmt=".1f",
                cmap="Blues",
                cbar=False,
                ax=ax,
                xticklabels=asses_list,
                yticklabels=asses_list,
            )
            ax.set_title(f"{machine}-{auc_dict[machine][0]:.2f}")
            ax.set_ylabel("True label (%)")
            ax.set_xlabel("Predicted label")

        fig.subplots_adjust(top=0.85, wspace=0.4)
        plt.show()

    # for RSNA dataset
    else:
        machine_list = ["Brand_A", "Brand_B"]
        prec_dict = {machine: [] for machine in machine_list}
        recall_dict = {machine: [] for machine in machine_list}

        for machine in machine_list:
            for fold in range(1, 6):
                test_auc, test_confmat = main(subset, classifier_dir, fold, config, pred_tag=machine,
                                              clamp_method=clamp_method)
                prec_dict[machine].append(test_auc)
                recall_dict[machine].append(test_confmat)

        # compute and print the average precision and recall
        for machine in machine_list:
            avg_precision = sum(prec_dict[machine]) / len(prec_dict[machine])
            avg_recall = sum(recall_dict[machine]) / len(recall_dict[machine])
            print(f"{machine}: Average Precision = {avg_precision:.4f}, Average Recall = {avg_recall:.4f}")
