import os
import shutil
from pickle import load
from tempfile import mkdtemp
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch as t
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import ToTensor
from tqdm import tqdm


class RamDisk:
    def __init__(self, dataset_path):
        # create a ram disk
        self.ram_dir = mkdtemp(dir="/dev/shm")
        print("Ram disk initialized: %s" % self.ram_dir)

        # unzip it to ram
        with ZipFile(os.path.join(dataset_path), "r") as zip_ref:
            file_list = zip_ref.namelist()
            for file in tqdm(file_list, desc="Unzip"):
                zip_ref.extract(file, self.ram_dir)

    def get(self):
        return self.ram_dir

    def free(self):
        # free up ram space
        shutil.rmtree(self.ram_dir)


class BatchSampler(Sampler):
    def __init__(self, dataset, label_index, batch_size):
        """
        Args:
            dataset (Dataset): Your dataset object.
            label_index (int): The index of the label to balance on (e.g., 3 for patho_label).
            batch_size (int): Total number of samples in one batch.
                              It must be divisible by the number of classes.
        """
        self.dataset = dataset
        self.label_index = label_index
        self.batch_size = batch_size

        # Group sample indices by class.
        self.class_to_indices = {}
        for idx in range(len(dataset)):
            label = dataset[idx][label_index].item()  # extract label as a Python number
            self.class_to_indices.setdefault(label, []).append(idx)

        # List of unique class labels (they don't need to start at 0)
        self.classes = list(self.class_to_indices.keys())
        self.n_classes = len(self.classes)

        if batch_size % self.n_classes != 0:
            raise ValueError("batch_size must be divisible by the number of classes.")
        self.samples_per_class = batch_size // self.n_classes

    def __iter__(self):
        # Shuffle the indices for each class at the start of each epoch.
        for cls in self.classes:
            np.random.shuffle(self.class_to_indices[cls])

        # Determine the number of batches based on the smallest class.
        num_batches = min(len(indices) for indices in self.class_to_indices.values()) // self.samples_per_class

        for i in range(num_batches):
            batch = []
            for cls in self.classes:
                start = i * self.samples_per_class
                end = start + self.samples_per_class
                batch.extend(self.class_to_indices[cls][start:end])
            # Optional: Shuffle the order of samples within the batch.
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        # Compute the number of batches based on the class with the fewest samples.
        return min(len(indices) for indices in self.class_to_indices.values()) // self.samples_per_class


class CustomSet(Dataset):
    def __init__(self, dataset_dir, folder, csv_file, transform=None, pred_tag="patho", clamp="full", disp_msg=True):
        assert clamp in ["full", "window"], "clamp must be 'full' or 'window'"

        self.dataset_dir = os.path.join(dataset_dir, folder)
        self.transform = transform or ToTensor()
        self.pred_tag = pred_tag
        self.clamp = clamp

        # read csv file
        if isinstance(csv_file, pd.DataFrame):
            self.df = csv_file
        else:
            self.df = pd.read_csv(csv_file)

        # Allowed manufacturer tags
        allowed_tags = ["Clearview", "Selenia", "Senograph", "Brand_A", "Brand_B", "Brand_C"]

        # Apply filtering only if pred_tag is not None
        if self.pred_tag is None:
            # No filtering applied
            pass
        elif isinstance(self.pred_tag, list):
            # Only filter using valid tags from the list
            valid_tags = [tag for tag in self.pred_tag if tag in allowed_tags]
            self.df = self.df[self.df["Manufacturer"].isin(valid_tags)].reset_index(drop=True)
        elif self.pred_tag in allowed_tags:
            self.df = self.df[self.df["Manufacturer"] == self.pred_tag].reset_index(drop=True)

        # create label mapping
        self.machine_map = {"Clearview": 0, "Selenia": 1, "Senograph": 2, "Brand_A": 0, "Brand_B": 1}

        # print dataset summary
        if disp_msg:
            print(f"Found {len(self.df)} samples for {self.pred_tag}.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = os.path.join(self.dataset_dir, self.df.iloc[idx, 0])

        # extract attributes
        filename = self.df.loc[idx, "File"]
        machine = self.df.loc[idx, "Manufacturer"]

        # extract attributes and convert to labels
        machine_label = t.tensor(self.machine_map[machine])
        patho_label = t.tensor(self.df.loc[idx, "Label"], dtype=t.int)

        # load image
        img_data = Image.open(file_name)

        # apply image transform
        img_data = self.transform(img_data)

        # load display window
        with open(file_name.replace("png", "pkl"), "rb") as f:
            disp_window = t.tensor(load(f), dtype=t.float32)

        # normalize
        wl = disp_window[0]
        wh = disp_window[1]

        img_full = t.clamp(img_data, 0, 1)
        img_clipped = t.clamp(img_data, wl, wh)
        img_clipped = (img_clipped - wl) / (wh - wl)

        return img_full, img_clipped, machine_label, patho_label
