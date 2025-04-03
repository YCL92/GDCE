# GDCE

Learning to Harmonize Cross-vendor X-ray Images by Non-linear Image Dynamics Correction

### Prepare required dataset

Run `download-embed.py` to download the data required by the experiment. Additionally, download the RSNA pneumonia detection dataset from this [link]([RSNA Pneumonia Detection Challenge (2018) | RSNA](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)).

After that, run `prepare-dataset.py` to pre-process the data, you can specify which dataset you will use by changing `dataset_dir` in `config.yaml`.

### Train downstream task model

For evaluation purposes only, skip this section and simply follow the instructions in the "test" section.

Run `train-patho.py` to train the pathology classifier (breast density classification or phenomina detection).

#### Train GDCE

For evaluation purposes only, skip this section and simply follow the instructions in the "test" section.

Specify the checkpoint folder of your pathology classifier obtained from the previous step, as well as the target scanner you want to align to the reference one. Below is an example:

```
machine_config = ["Clearview", "embed-patho-full-20250317_143344"]
```

and then run `train-enhance.py` to train GDCE. 

#### Test and compare results

Replace the default GDCE and downstream task model checkpoint folders. Below is an example:

```
enhancer_dir = "embed-clearview-20250308_010625" 
classifier_dir = "embed-patho-full-20250307_164235"
```

Run `test-patho.py` and `test-gdce.py` to get the 5-fold cross-validation results without and with GDCE normalization.