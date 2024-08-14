# DFUC 2024: Self-supervised Instance Segmentation of Diabetic Foot Ulcers

This work is a part of submission made to [DFUC 2024](https://dfu-challenge.github.io/).
DFUC 2024 is hosted by [MICCAI 2024](https://conferences.miccai.org/2024/en/), the 27th International Conference on Medical Image Computing and Computer Assisted Intervention.

## Setting Up a Virtual Environment

### Create a virtual environment and activate
```bash
conda create -n dfuc python=3.8 -y
````
```bash
conda activate dfuc
````

### Install required modules
```bash
pip install xformers
````
```bash
conda install torchvision
````
```bash
pip install matplotlib, tqdm, opencv-python
````
```bash
pip install jupyter jupyterlab
````

## Train
<li> Please refer to the batch training in <b>batch_train.ipynb</b>.</li>

## Scores
<li> You can calculate Dice similarity coefficient(DSC), Intersect over Union (IoU), False Negative Error (FNE) and False Positive Error (FPE) in <b>score.ipynb</b>.</li>

## Predicted Masks
<li> You can save the predicted masks using <b>save_masks.ipynb</b>.</li>
