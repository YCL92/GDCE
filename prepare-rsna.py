import ast
import os

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm


def extract_dicom_metadata(dicom_dir):
    file_list = []
    for root, dirs, files in os.walk(dicom_dir):
        for filename in files:
            if filename.lower().endswith(".dcm"):
                file_list.append(os.path.join(root, filename))

    metadata_list = []
    for dicom_path in tqdm(file_list, desc="Processing DICOM files"):
        try:
            ds = pydicom.dcmread(dicom_path)
            file_metadata = {}

            # extract each metadata element
            for data_element in ds:
                if data_element.keyword == "PixelData":
                    continue
                key = data_element.keyword if data_element.keyword else str(data_element.tag)
                file_metadata[key] = str(data_element.value)

            # store the file path
            file_metadata["dicom_filepath"] = dicom_path

            metadata_list.append(file_metadata)
        except Exception as e:
            print(f"Could not read DICOM file {dicom_path}: {e}")

    df = pd.DataFrame(metadata_list)

    return df


def parse_first_pixel_spacing(val):
    if not isinstance(val, str):
        return None
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
            return float(parsed[0])
    except Exception:
        pass
    return None


def merge_pixel_spacing(value):
    if value is None:
        return None

    if np.isclose(value, 0.171, atol=5e-4):
        return "Brand_A"
    if np.isclose(value, 0.143, atol=5e-4):
        return "Brand_B"
    if np.isclose(value, 0.1943, atol=1e-3):
        return "Brand_C"
    return None


def merge_target_labels(df, label_csv):
    annotation_df = pd.read_csv(label_csv)
    annotation_agg = annotation_df.groupby("patientId", as_index=False)["Target"].max()

    # rename "PatientID" to "patientId" in the metadata for merging
    df_renamed = df.rename(columns={"PatientID": "patientId"})
    df_merged = pd.merge(df_renamed, annotation_agg, on="patientId", how="left")

    # fill missing Target values with 0 and convert to int
    df_merged["Target"] = df_merged["Target"].fillna(0).astype(int)
    df_merged.rename(columns={'Target': 'PathoLabel'}, inplace=True)

    return df_merged


if __name__ == "__main__":
    # extract or load DICOM metadata
    dicom_folder_path = "/mnt/share/data/RSNA-Pheumonia/stage_2_train_images"  # <-- update this path accordingly
    df = extract_dicom_metadata(dicom_folder_path)

    # process PixelSpacing, parse and merge similar groups
    df["Pixel Spacing"] = df["PixelSpacing"].apply(parse_first_pixel_spacing)
    df["ManufacturerModelName"] = df["Pixel Spacing"].apply(merge_pixel_spacing)
    df_filtered = df[df["ManufacturerModelName"].notna()].copy()

    # merge annotation labels from the CSV file
    label_csv_path = "/mnt/share/data/RSNA-Pheumonia/stage_2_train_labels.csv"
    df_merged = merge_target_labels(df_filtered, label_csv_path)

    # for each scanner and each Target, randomly select up to max_num_sample samples
    max_num_sample = 500
    sampled_df = (
        df_merged.groupby(["ManufacturerModelName", "PathoLabel"], group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), max_num_sample), random_state=42))
        .reset_index(drop=True)
    )

    # save the sampled dataframe instead of the full merged dataframe
    sampled_df.to_csv("./data/processed-rsna.csv", index=False)

    # print the target value counts for each grouped pixel spacing
    print("\nTarget counts for each merged PixelSpacing group in the sampled dataframe:")
    grouped_target_counts = sampled_df.groupby("ManufacturerModelName")["PathoLabel"].value_counts()
    print(grouped_target_counts)
