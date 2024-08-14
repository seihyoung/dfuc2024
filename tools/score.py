import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from tools.dataset import  MaskRCNNDataset
from tqdm.notebook import tqdm

def resized(gt, pred):
    max_height = max(gt.shape[0], pred.shape[0])
    max_width = max(gt.shape[1], pred.shape[1])
  
    resized_gt = np.zeros((max_height, max_width), dtype=bool)
    resized_pred = np.zeros((max_height, max_width), dtype=bool)
    
    resized_gt[:gt.shape[0], :gt.shape[1]] = gt
    resized_pred[:pred.shape[0], :pred.shape[1]] = pred

    return resized_gt, resized_pred


def dice(gt, pred, smooth=1e-6):   # Dice Similarity coefficient
    resized_gt, resized_pred = resized(gt, pred)
    
    tp = np.sum(np.logical_and(resized_gt == 1, resized_pred == 1))
    fp = np.sum(np.logical_and(resized_gt == 0, resized_pred == 1))
    fn = np.sum(np.logical_and(resized_gt == 1, resized_pred == 0))
    
    dsc = ((2 * tp + smooth) / (2*tp + fp + fn + smooth))
    
    return dsc


def iou(gt, pred):  # Intersect over union
    resized_gt, resized_pred = resized(gt, pred)
    
    intersection = np.sum(np.logical_and(resized_gt, resized_pred))
    union = np.sum(np.logical_or(resized_gt, resized_pred))
    
    if union != 0:
        iou = intersection/union
    else:
        iou = 0.
    
    return iou


def fne(gt, pred):   # False Negative Error
    resized_gt, resized_pred = resized(gt, pred)
    fn = np.sum(np.logical_and(resized_gt == 1, resized_pred == 0))
    tp = np.sum(np.logical_and(resized_gt == 1, resized_pred == 1))
    return fn / (fn + tp)


def fpe(gt, pred):   # False Positive Error
    resized_gt, resized_pred = resized(gt, pred)
    tn = np.sum(np.logical_and(resized_gt == 0, resized_pred == 0))
    fp = np.sum(np.logical_and(resized_gt == 0, resized_pred == 1))
    return fp / (fp + tn)