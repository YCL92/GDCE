import os
import subprocess

import pandas as pd
import yaml
from tqdm import tqdm

# load the dataframe from the CSV file
csv_file_path = "./data/processed-embed.csv"

# AWS profile
aws_profile = "embed-profile"

# AWS S3 base URL
s3_bucket_url = "s3://embed-dataset-open/images"

# load config file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# base directory
base_dst_dir = os.path.join(config["dataset_dir"], "images")

# ensure the base dst directory exists
os.makedirs(base_dst_dir, exist_ok=True)

# loop over all samples
df_sampled = pd.read_csv(csv_file_path)
for path in tqdm(df_sampled["anon_dicom_path"], desc="Downloading files", unit="file"):
    # full S3 path
    s3_file_path = os.path.join(s3_bucket_url, path)

    # local destination path
    local_destination_path = os.path.join(base_dst_dir, path)

    # check if file already exists
    if os.path.exists(local_destination_path):
        tqdm.write(f"File {path} already exists, skipping.")
        continue
    else:
        local_dir = os.path.dirname(local_destination_path)
        os.makedirs(local_dir, exist_ok=True)

    # construct the AWS CLI command
    command = ["aws", "s3", "cp", s3_file_path, local_destination_path, "--profile", aws_profile]

    # execute the command and suppress output
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        tqdm.write(f"Error downloading {path}: {e}")
