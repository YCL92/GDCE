import re
from os.path import join

import pandas as pd
import yaml

# user defined param
max_num_sample = 500

# Load config file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# load csv files
meta_df = pd.read_csv(join(config["dataset_dir"], "tables/EMBED_OpenData_metadata.csv"), low_memory=False)[
    [
        "empi_anon",
        "acc_anon",
        "anon_dicom_path",
        "SeriesDescription",
        "FinalImageType",
        "ImageLateralityFinal",
        "ViewPosition",
        "spot_mag",
        "PatientSex",
        "Manufacturer",
        "ManufacturerModelName",
    ]
]
clinical_df = pd.read_csv(join(config["dataset_dir"], "tables/EMBED_OpenData_clinical.csv"), low_memory=False)[
    ["empi_anon", "acc_anon", "tissueden", "asses"]
].drop_duplicates(subset="acc_anon")

# process ViewPosition
meta_df.loc[(meta_df["SeriesDescription"] == "RXCCL") | (meta_df["SeriesDescription"] == "LXCCL"), "ViewPosition"] = (
    "XCCL"
)
view_nan = meta_df.loc[(meta_df.ViewPosition.isna()) & (meta_df.SeriesDescription.notna())]
dicom_no_nans = meta_df[~meta_df.index.isin(view_nan.index)]
view_nan["ViewPosition"] = view_nan["SeriesDescription"].apply(
    lambda x: "CC" if "CC" in x else ("MLO" if "MLO" in x else None)
)
meta_df = pd.concat([dicom_no_nans, view_nan], axis=0, ignore_index=True)
meta_df = meta_df.drop_duplicates(subset="anon_dicom_path")

# data cleaning
# meta_df = meta_df[meta_df.ImageLateralityFinal == "R"]
meta_df = meta_df[meta_df.spot_mag.isna()]
meta_df = meta_df[meta_df.ViewPosition == "CC"]
meta_df = meta_df[meta_df.FinalImageType == "2D"]
meta_df = meta_df[meta_df.PatientSex == "F"]
meta_df = meta_df[
    meta_df.ManufacturerModelName.isin(
        ["Selenia Dimensions", "Clearview CSm", "Senograph 2000D ADS_17.4.5", "Senograph 2000D ADS_17.5"]
    )
]
meta_df["ManufacturerModelName"] = meta_df["ManufacturerModelName"].replace(
    r"Senograph 2000D ADS_.*", "Senograph 2000D", regex=True
)

# remove prefix and duplicates
meta_df["anon_dicom_path"] = meta_df["anon_dicom_path"].apply(lambda path: re.sub(r"^.*(cohort_[12]/)", r"\1", path))
meta_df = meta_df.drop_duplicates(subset="anon_dicom_path")

# merge metadata and clinical data
merged_df = clinical_df.merge(meta_df, on=["empi_anon", "acc_anon"])
merged_df["PathoLabel"] = merged_df["tissueden"].map({1.0: 0, 2.0: 1, 3.0: 2, 4.0: 3}).fillna(-1).astype(int)

# generate subset for pathology experiment
df_sampled = pd.DataFrame()
for manufacturer, m_group in merged_df.groupby("ManufacturerModelName"):
    manufacturer = manufacturer.split(" ")[0]
    m_group = m_group[m_group.tissueden.isin([1, 2, 3, 4])]

    for tissueden, a_group in m_group.groupby("tissueden"):
        unique_empi_anons = a_group["empi_anon"].unique()

        # sample n_sample unique empi_anons
        sample_count = min(max_num_sample, len(unique_empi_anons))
        sampled_empi_anons = pd.Series(unique_empi_anons).sample(sample_count, random_state=42)

        # sample one anon_dicom_path from each empi_anon
        for empi in sampled_empi_anons:
            # get the subset for this empi_anon
            subset = a_group[a_group["empi_anon"] == empi]

            # randomly select one anon_dicom_path
            selected_row = subset.sample(n=1, random_state=42)
            selected_row["ManufacturerModelName"] = manufacturer

            # append to the final result
            df_sampled = pd.concat([df_sampled, selected_row], ignore_index=True)

# generate subset for machine experiment
for manufacturer, m_group in df_sampled.groupby("ManufacturerModelName"):
    manufacturer = manufacturer.split(" ")[0]
    print(manufacturer)
    print(m_group.tissueden.value_counts())

# write to file
df_sampled.to_csv("./data/processed-embed.csv", index=False)
