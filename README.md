# Dinov2_Mask R-CNN: Self-supervised Instance Segmentation of Diabetic Foot Ulcers

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
## Preparation of the Datasets
For fine-tuning, we utilized the DFUC2022 dataset, which includes both images and corresponding masks. The dataset was randomly split into two subsets: 80% of the data was used for training, while the remaining 20% was reserved for validation. To ensure reproducibility in model development, we pre-processed and prepared the augmented data beforehand, rather than applying augmentation dynamically during training. The key parameters for the augmentation process are outlined below. Notably, aspects such as color and brightness were not considered in this augmentation strategy.
````bash
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomAffine(degrees=25, scale=(0.7, 1.3)),
])
````

## How to train
<li> Please refer to the batch training in <b>batch_train.ipynb</b>.</li>
<li> The <b>save_epoch_freq</b> option is employed to save the intermediate model results at every specified epoch frequency. </li>

## How to get scores
<li> You can calculate Dice similarity coefficient(DSC), Intersect over Union (IoU), False Negative Error (FNE) and False Positive Error (FPE) in <b>score.ipynb</b>.</li>

## How to predict masks and save
<li> You can save the predicted masks using <b>save_masks.ipynb</b>.</li>

# Used networks
The following two models were used.
<li><a href="https://github.com/facebookresearch/dinov2">Dinov2</li>
<li><a href="https://github.com/matterport/Mask_RCNN">Mask R-CNN</li>

# Acknowledgements
This work was supported by the Institute of Information & communications Technology Planning & Evaluation(IITP) grant funded by the Korea government(MSIT) (No. 2022-0-00682, Development of XR twin technology for the management of chronic skin diseases in the elderly)

