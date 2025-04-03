import copy
import csv
import os
from random import seed as radnseed

import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numpy.random import seed as npseed
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import manual_seed as tseed, save
from torch.cuda import manual_seed, manual_seed_all


def initRandomSeed(seed=42):
    tseed(seed)
    radnseed(seed)
    npseed(seed)
    manual_seed_all(seed)
    manual_seed(seed)


class ModelSaver:

    def __init__(self, mode="min", save_dir="../checkpoint", file_name=None):
        assert mode in ["min", "max"], f'mode should be either "min" or "max"'

        self.sign = 1 if mode == "min" else -1
        self.save_dir = save_dir
        self.file_name = file_name
        self.best_loss = np.Inf
        self.best_model = None
        self.best_epoch = None

    def step(self, cur_loss, model, epoch):
        signed_loss = self.sign * cur_loss
        if signed_loss <= self.best_loss:
            self.best_loss = signed_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            save_ckpt = True
        else:
            save_ckpt = False

        # save model if needed
        if save_ckpt:
            if self.file_name:
                save_path = os.path.join(self.save_dir, self.file_name)
            else:
                save_path = os.path.join(self.save_dir, "checkpoint-%03d.pt" % epoch)
            save(self.best_model, save_path)
            print("New checkpoint saved.")

    def get(self):
        return self.best_model, self.best_epoch


def write2CSV(filename, data):
    # check if the file exists
    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as csvfile:
        fieldnames = data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # write header if the file doesn't exist
        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


def centeredSubplots(nrows, ncols, figsize, num_figs):
    fig = plt.figure(figsize=figsize)
    axs = []

    m = num_figs % ncols
    m = range(1, ncols + 1)[-m]  # subdivision of columns
    gs = gridspec.GridSpec(nrows, m * ncols)

    for i in range(0, num_figs):
        row = i // ncols
        col = i % ncols

        if row == nrows - 1:  # center only last row
            off = int(m * (ncols - num_figs % ncols) / 2)
        else:
            off = 0

        ax = plt.subplot(gs[row, m * col + off : m * (col + 1) + off])
        axs.append(ax)

    return fig, axs


def makeParts(file_dict, train_samples):
    # calculate the total number of True and False samples
    scan_labels = {}

    for scan_id, files in file_dict.items():
        true_count = sum("pos" in file for file in files)
        false_count = len(files) - true_count
        scan_labels[scan_id] = {"true": true_count, "false": false_count}

    # sort scans based on the number of samples they contain
    sorted_scans = sorted(scan_labels.items(), key=lambda x: sum(x[1].values()), reverse=True)

    # distribute the scans into two partitions
    part1 = {"file_list": [], "scan_list": [], "num_pos": 0, "num_neg": 0}
    part2 = {"file_list": [], "scan_list": [], "num_pos": 0, "num_neg": 0}

    for scan_id, counts in sorted_scans:
        if len(part1["file_list"]) + sum(counts.values()) <= train_samples:
            part1["file_list"].extend(file_dict[scan_id])
            part1["scan_list"].append(scan_id)
            part1["num_pos"] += counts["true"]
            part1["num_neg"] += counts["false"]

        else:
            part2["file_list"].extend(file_dict[scan_id])
            part2["scan_list"].append(scan_id)
            part2["num_pos"] += counts["true"]
            part2["num_neg"] += counts["false"]

    return part1, part2


def splitDict(in_dict, num_samples):
    total_samples = sum(len(samples) for samples in in_dict.values())
    keys = list(in_dict.keys())
    split_samples = 0
    split_index = 0

    for i, key in enumerate(keys):
        split_samples += len(in_dict[key])
        if split_samples >= num_samples:
            split_index = i + 1
            break

    keys_p1 = keys[:split_index]
    keys_p2 = keys[split_index:]

    dict_p1 = {key: in_dict[key] for key in keys_p1}
    dict_p2 = {key: in_dict[key] for key in keys_p2}

    return dict_p1, dict_p2


def cmpMetrics(meta_data, target_fold):
    filtered_data = [row for row in meta_data if row[0] == int(target_fold)]

    if not filtered_data:
        print(f"No data found for fold {int(target_fold)}.")
        return None, None

    # extract ground-truth and predicted labels
    gt_list = [row[2] for row in filtered_data]
    pred_list = [row[4] for row in filtered_data]

    # ompute ROC-AUC score
    roc_auc = roc_auc_score(gt_list, pred_list)

    # ompute confusion matrix
    conf_matrix = confusion_matrix(gt_list, pred_list)

    return roc_auc, conf_matrix


def plotConfMat(confusion_matrix, class_names, avg_auc, save_name=None, figsize=(4, 3)):
    plt.figure(figsize=figsize)
    sns.heatmap(
        confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=True, xticklabels=class_names, yticklabels=class_names
    )
    plt.title("Avg AUC: %.2f" % avg_auc)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    if save_name:
        plt.savefig(save_name)

    plt.show()
