import os
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset
from tools.dataset import  MaskRCNNDataset
from tools.functions import ssl_maskrcnn_train, ssl_maskrcnn_validation, ssl_maskrcnn_infer


BACKBONE_OUT_DIMS = 1024
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def visualize_image(model, in_image, gt = None, mask_threshold = 0.5, score_threshold = 0.2, img_transform=None, save_plt = False):
    img = Image.open(in_image).convert('RGB').resize((BACKBONE_OUT_DIMS, BACKBONE_OUT_DIMS))
    transformed_image, preds = ssl_maskrcnn_infer(model, device, in_image, img_transform = img_transform)
    if gt is None:
        masks = np.zeros((BACKBONE_OUT_DIMS, BACKBONE_OUT_DIMS))
    else:
        masks = Image.open(gt).resize((BACKBONE_OUT_DIMS, BACKBONE_OUT_DIMS))

    print('scores=', preds['scores'])
    all_preds_masks = np.zeros((BACKBONE_OUT_DIMS, BACKBONE_OUT_DIMS))
    for index, mask in enumerate(preds['masks'].cpu().detach().numpy()):
        if type(score_threshold) == list:
            if score_threshold[0] <= preds['scores'][index] <= score_threshold[1]:
                all_preds_masks = np.logical_or(all_preds_masks, mask[0] > mask_threshold)
        else:
            if preds['scores'][index] > score_threshold:
                all_preds_masks = np.logical_or(all_preds_masks, mask[0] > mask_threshold) 
    xor_masks = np.logical_xor(masks, all_preds_masks)
        
    fig, ax = plt.subplots(1, 4, figsize=(16,4))
    ax[0].imshow(img)
    ax[0].set_title('Original Image')
    ax[1].imshow(masks)
    ax[1].set_title('Mask GT Image')    
    ax[2].imshow(all_preds_masks)
    ax[2].set_title('Mask Predication Image')        
    ax[3].imshow(xor_masks)
    ax[3].set_title('Mask_Pred XoR Image')  
    
    if save_plt is True:
        filename = in_image.split('.')[0] + '_result.png'        
        plt.savefig(filename)
        print('Saved result in ' + filename)
    else:
        plt.show()


def save_masks(model, img_dir, out_dir, mask_threshold = 0.5, score_threshold = 0.2, alpha = 0.6, img_transform=None):
    print(f'mask threshold={mask_threshold}, score threshold={score_threshold}, alpha={alpha}')
    
    pred_mask_path = os.path.join(out_dir, 'pred_mask')
    overlapped_path = os.path.join(out_dir, 'overlapped')
    os.makedirs(pred_mask_path, exist_ok=True)
    os.makedirs(overlapped_path, exist_ok=True)
    
    filelist = tqdm(os.listdir(img_dir))
    for f in enumerate(filelist):
        file = f[1] # filename
        if not file.lower().endswith(('.jpg', '.png')):
            continue
        file_path = os.path.join(img_dir, file)
        img = cv2.imread(file_path)
        height, width, ch = img.shape
        transformed_image, preds = ssl_maskrcnn_infer(model, device, file_path, img_transform)

        
        pred_mask_filename = os.path.join(pred_mask_path, file.split('.')[0] + '.png')
        all_preds_masks = np.zeros((BACKBONE_OUT_DIMS, BACKBONE_OUT_DIMS))
        
        for index, mask in enumerate(preds['masks'].cpu().detach().numpy()):
            if (preds['scores'][index] > score_threshold):
                all_preds_masks = np.logical_or(all_preds_masks, mask[0] > mask_threshold)
        
        pred_mask = cv2.resize(all_preds_masks.astype(np.uint8), (width,height), interpolation=cv2.INTER_LINEAR)
        pred_mask = pred_mask * 255
        grayMask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(pred_mask_filename, pred_mask) # 8bit
    
        # overlapped image 저장
        
        overlapped_filename = os.path.join(overlapped_path, file.split('.')[0] + '_overlapped.png')
        overlapped_image = cv2.addWeighted(img, alpha, grayMask, (1-alpha), 0)
        cv2.imwrite(overlapped_filename, overlapped_image)
    print('Masks are saved in directory ' + out_dir)
