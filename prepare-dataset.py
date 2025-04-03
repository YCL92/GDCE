import csv
import random
from collections import defaultdict
from os import makedirs
from os.path import join
from pickle import dump

import cv2
import numpy as np
import pandas as pd
import yaml
from pydicom import dcmread
from pydicom.multival import MultiValue
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from util.img_util import equalizeHist


def process_dicom(r_idx, dcm_row, img_size, data_dir, data_save_dir, norm="full"):
    assert norm in ["full", "window", "hist"]

    # generate output metadata
    out_meta = {
        "manufacturer": dcm_row["ManufacturerModelName"],
        "patho_label": dcm_row["PathoLabel"],
        "filename": None,
    }

    # read dicom
    if "anon_dicom_path" in df.columns:
        dcm_path = join(data_dir, "images", dcm_row["anon_dicom_path"])
        dcm = dcmread(dcm_path)
        pixel_log = dcm.PixelIntensityRelationship
        orientation = dcm.PatientOrientation[0]
        crop_flag = True
    else:
        dcm_path = dcm_row["dicom_filepath"]
        dcm = dcmread(dcm_path)
        pixel_log = "LOG"
        orientation = dcm.PatientOrientation
        crop_flag = False

    # select only "LOG" relation
    if pixel_log != "LOG":
        return None

    img_fname = f"resize-{r_idx + 1:04d}.png"
    out_meta["filename"] = img_fname

    # # flip if needed
    # if orientation == "A":
    #     pixel_array = np.fliplr(dcm.pixel_array)
    # else:
    #     pixel_array = dcm.pixel_array
    #
    # # extract display window
    # try:
    #     window_center = dcm.WindowCenter
    #     window_width = dcm.WindowWidth
    # except AttributeError as e:
    #     window_center = int((2 ** dcm.BitsStored - 1) / 2)
    #     window_width = int(2 ** dcm.BitsStored - 1)
    #
    # if isinstance(window_center, (MultiValue, list)):
    #     window_center = float(window_center[0])
    # else:
    #     window_center = float(window_center)
    #
    # if isinstance(window_width, (MultiValue, list)):
    #     window_width = float(window_width[0])
    # else:
    #     window_width = float(window_width)
    #
    # # check photometric interpretation (MONOCHROME1 needs inversion)
    # photometric_interpretation = dcm.PhotometricInterpretation
    # if photometric_interpretation == "MONOCHROME1":
    #     pixel_array = (2 ** dcm.BitsStored - 1) - pixel_array
    #
    # # for EMBED data, crop to valid ROI
    # if crop_flag:
    #     img_8bit = (pixel_array.astype(float) / (2 ** dcm.BitsStored - 1) * 255).astype(np.uint8)
    #
    #     # extract breast contour
    #     blur_img = cv2.GaussianBlur(img_8bit, (5, 5), 0)
    #     _, binary = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     largest_contour = max(contours, key=cv2.contourArea)
    #
    #     # filter points within the 10%-90% y-axis range
    #     contour_points = largest_contour[:, 0, :]
    #     filtered_points = contour_points[
    #         (contour_points[:, 1] >= int(0.1 * img_8bit.shape[0]))
    #         & (contour_points[:, 1] <= int(0.9 * img_8bit.shape[0]))
    #         ]
    #
    #     # crop to the contour
    #     crop_x, _, crop_w, _ = cv2.boundingRect(filtered_points)
    #     _, crop_y, _, crop_h = cv2.boundingRect(contour_points)
    #     cropped_img = pixel_array[crop_y: crop_y + crop_h, crop_x:]
    #
    # else:
    #     cropped_img = pixel_array
    #
    # # normalize according to the selected method
    # if norm == "full":
    #     normalized_img = (cropped_img.astype(float) / (2 ** dcm.BitsStored - 1) * 65535).astype(np.uint16)
    #
    # elif norm == "window":
    #     window_min = window_center - window_width / 2
    #     normalized_img = (cropped_img - window_min) / window_width * 65535
    #     normalized_img = np.clip(normalized_img, 0, 65535).astype(np.uint16)
    #
    # elif norm == "hist":
    #     normalized_img = (cropped_img.astype(float) / (2 ** dcm.BitsStored - 1) * 65535).astype(np.uint16)
    #     normalized_img = equalizeHist(normalized_img, largest_contour)
    #
    # # determine the scaling factor
    # h, w = normalized_img.shape
    # if h > w:
    #     scale = img_size / h
    #     new_h, new_w = img_size, int(w * scale)
    # else:
    #     scale = img_size / w
    #     new_h, new_w = int(h * scale), img_size
    #
    # # resize the image while maintaining aspect ratio
    # resized_img = cv2.resize(normalized_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    #
    # # save image
    # img_fname = f"resize-{r_idx + 1:04d}.png"
    # out_meta["filename"] = img_fname
    # cv2.imwrite(join(data_save_dir, img_fname), resized_img)
    #
    # # save display window
    # wl = np.clip((window_center - window_width / 2) / (2 ** dcm.BitsStored - 1), 0, 1)
    # wh = np.clip((window_center + window_width / 2) / (2 ** dcm.BitsStored - 1), 0, 1)
    # disp_fname = "resize-%04d.pkl" % (r_idx + 1)
    # with open(join(data_save_dir, disp_fname), "wb") as f:
    #     dump([wl, wh], f)

    return out_meta


def kfoldSplit(data, n_split, random_state=42):
    labels = [f"{row['manufacturer']}-{row['patho_label']}" for row in data]
    skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=random_state)

    splits = []
    for train_idx, val_idx in skf.split(data, labels):
        train_split = [data[i] for i in train_idx]
        val_split = [data[i] for i in val_idx]
        splits.append((train_split, val_split))

    return splits


if __name__ == "__main__":
    # user defined params
    norm_type = "full"
    write_csv = True

    # init random seed
    random.seed(42)

    # read setup files
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    if "EMBED" in config["dataset_dir"]:
        csv_file_path = "./data/processed-embed.csv"
    else:
        csv_file_path = "./data/processed-rsna.csv"

    df = pd.read_csv(csv_file_path)

    # make folder for data
    data_save_dir = join(config["dataset_dir"], f"images-{norm_type}_{config['img_size']}")
    makedirs(data_save_dir, exist_ok=True)

    # process all samples
    meta_list = []
    for r_idx, df_row in tqdm(df.iterrows(), total=len(df), desc="Processing DICOM files"):
        # process each DICOM file sequentially
        img_meta = process_dicom(
            int(r_idx), df_row, config["img_size"], config["dataset_dir"], data_save_dir, norm_type
        )

        # append metadata if the DICOM file is valid
        if img_meta is not None:
            meta_list.append(img_meta)

    if write_csv:
        # make folder for csv files
        if "EMBED" in config["dataset_dir"]:
            csv_save_dir = "data/embed-5fold-csv"
        else:
            csv_save_dir = "data/rsna-5fold-csv"
        makedirs(csv_save_dir, exist_ok=True)

        # group meta_list by manufacturer
        grouped_meta = defaultdict(list)
        for img_meta in meta_list:
            menufacturer = img_meta["manufacturer"]
            patho_label = img_meta["patho_label"]
            grouped_meta[f"{menufacturer}-{patho_label}"].append(img_meta)

        # split to train/val/test sets
        if "EMBED" in config["dataset_dir"]:
            train_list = (
                grouped_meta["Clearview-1"][:400]
                + grouped_meta["Clearview-2"][:400]
                + grouped_meta["Senograph-1"][:400]
                + grouped_meta["Senograph-2"][:400]
                + grouped_meta["Selenia-0"][:400]
                + grouped_meta["Selenia-1"][:400]
                + grouped_meta["Selenia-2"][:400]
                + grouped_meta["Selenia-3"][:400]
            )
            test_list = (
                grouped_meta["Clearview-0"]
                + grouped_meta["Clearview-3"]
                + grouped_meta["Clearview-1"][400:]
                + grouped_meta["Clearview-2"][400:]
                + grouped_meta["Selenia-0"][400:]
                + grouped_meta["Selenia-1"][400:]
                + grouped_meta["Selenia-2"][400:]
                + grouped_meta["Selenia-3"][400:]
                + grouped_meta["Senograph-1"][400:]
                + grouped_meta["Senograph-2"][400:]
                + grouped_meta["Senograph-0"]
                + grouped_meta["Senograph-3"]
            )

        else:
            train_list = (
                grouped_meta["Brand_A-0"][:400]
                + grouped_meta["Brand_A-1"][:400]
                + grouped_meta["Brand_B-0"][:400]
                + grouped_meta["Brand_B-1"][:400]
            )
            test_list = (
                grouped_meta["Brand_A-0"][400:]
                + grouped_meta["Brand_A-1"][400:]
                + grouped_meta["Brand_B-0"][400:]
                + grouped_meta["Brand_B-1"][400:]
            )

        # print statistics
        print(f"Total training samples: {len(train_list)}")
        print(f"Total testing samples: {len(test_list)}")

        # stratified 5-fold splitting
        stratified_split = kfoldSplit(train_list, n_split=5)
        for fold_idx, (train_fold, val_fold) in enumerate(stratified_split):
            # write training fold to CSV
            train_filename = join(csv_save_dir, f"train-fold{fold_idx + 1}.csv")
            with open(train_filename, mode="w", newline="") as train_file:
                writer = csv.writer(train_file)
                writer.writerow(["File", "Manufacturer", "Label"])
                writer.writerows([[row["filename"], row["manufacturer"], row["patho_label"]] for row in train_fold])

            # write validation fold to CSV
            val_filename = join(csv_save_dir, f"val-fold{fold_idx + 1}.csv")
            with open(val_filename, mode="w", newline="") as val_file:
                writer = csv.writer(val_file)
                writer.writerow(["File", "Manufacturer", "Label"])
                writer.writerows([[row["filename"], row["manufacturer"], row["patho_label"]] for row in val_fold])

        # create test set
        test_filename = join(csv_save_dir, "test.csv")
        with open(test_filename, mode="w", newline="") as test_file:
            writer = csv.writer(test_file)
            writer.writerow(["File", "Manufacturer", "Label"])
            writer.writerows([row["filename"], row["manufacturer"], row["patho_label"]] for row in test_list)
