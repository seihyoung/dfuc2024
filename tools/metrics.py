import torch
import numpy as np
from tqdm.notebook import tqdm
from tools.functions import ssl_maskrcnn_infer
from tools.score import dice, iou, fne, fpe

# get dice value from model and dataset
BACKBONE_OUT_DIMS = 1024
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_metrics(model, data_loader, mask_threshold = 0.5, score_threshold = 0.2):
    print(f'mask_threshold={mask_threshold}, score_threshold={score_threshold}')
    
    value_dsc = 0.
    value_iou = 0.
    value_fne = 0.
    value_fpe = 0.
    
    data_number = len(data_loader)
    print(f'Loaded data number: {data_number}')
    
    loop = tqdm(range(data_number), desc='Get dice progress', total=data_number, leave=False)
    for data_idx in loop:
        image, target = data_loader[data_idx]
        file_idx = str(target['image_id'].item())
        
        transformed_image, preds = ssl_maskrcnn_infer(model, device, image=image)
                
        masks = np.zeros((BACKBONE_OUT_DIMS, BACKBONE_OUT_DIMS))   
        for gt_idx, mask in enumerate(target['masks']):
            masks = np.logical_or(masks, mask)
               
        all_preds_masks = np.zeros((BACKBONE_OUT_DIMS, BACKBONE_OUT_DIMS))    
        for pred_idx, pred_mask in enumerate(preds['masks'].cpu().detach().numpy()):
            if (preds['scores'][pred_idx] > score_threshold):
                all_preds_masks = np.logical_or(all_preds_masks, pred_mask[0] > mask_threshold)    
                        
        value_dsc += dice(masks, all_preds_masks, smooth=1e-6)/data_number
        value_iou += iou(masks, all_preds_masks)/data_number
        value_fne += fne(masks, all_preds_masks)/data_number
        value_fpe += fpe(masks, all_preds_masks)/data_number
        
        loop.set_postfix(dsc=f'{value_dsc:.4f}', iou=f'{value_iou:.4f}', fne=f'{value_fne:.4f}', fpe=f'{value_fpe:.4f}')
            
    return {'dsc': value_dsc, 'iou': value_iou, 'fne': value_fne, 'fpe': value_fpe}