# GDCE

Learning to Harmonize Cross-vendor X-ray Images by Non-linear Image Dynamics Correction

### Prepare required dataset

Run `download-embed.py` to download the data required by the experiment, and run `prepare-embed.py` to process the EMBED dataset.

Additionally, download the RSNA pneumonia detection dataset from this [link]([RSNA Pneumonia Detection Challenge (2018) | RSNA](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)), and run `prepare-rsna.py` to process the RSNA dataset.

After that, run `prepare-dataset.py` to pre-process the data, you can specify which dataset you will use by changing `dataset_name`.

### Train downstream task model

For evaluation purposes only, skip this section and simply follow the instructions in the "test" section. You can download our pre-trained weights from this [link](https://drive.google.com/drive/folders/1MvSS2VNVg7R2f-2aEkxLmjpk3zHOHMQ9?usp=sharing) and unzip to the "checkpoint" folder.

Run `train-patho.py` to train the pathology classifier (breast density classification or phenomina detection).

#### Train GDCE

For evaluation purposes only, skip this section and simply follow the instructions in the "test" section.

Specify the checkpoint folder of your pathology classifier obtained from the previous step, as well as the target scanner you want to align to the reference one. Below is an example:

```
machine_config = ["Clearview", "embed-patho-full-20250317_143344"]
```

and then run `train-enhance.py` to train GDCE. 

#### Test and compare results

<u>Test without GDCE: </u>Replace the  downstream task model checkpoint folder. Below is an example:

```
classifier_dir = 'embed-patho-full-20250317_143344'
```

Run `test-enhance.py` to get the 5-fold cross-validation results.

<u>Test with GDCE:</u> Replace the default GDCE and downstream task model checkpoint folders. Below is an example:

```
machine_config = ["Clearview", "embed-clearview-20250406_142325", "embed-patho-full-20250317_143344"]
```

Run `test-enhance.py` to get the 5-fold cross-validation results.