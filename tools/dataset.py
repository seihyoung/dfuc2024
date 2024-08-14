import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import collections
import glob
import cv2


class MaskRCNNDataset(Dataset):
    def __init__(self, img_dir, mask_dir, num_classes, img_transform=None, mask_transform=None, image_info=None, resize=False):
        self.num_classes = num_classes
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.image_dir = img_dir
        self.mask_dir = mask_dir

        if image_info is None:
            self.image_info = collections.defaultdict(dict)
            img_list = sorted(glob.glob(self.image_dir+'/*.jpg'))
            mask_list = sorted(glob.glob(self.mask_dir+'/*.png'))
            for index in range(len(img_list)):
                file_idx = img_list[index].split('/')[-1].split('.')[0]
                self.image_info[index] = {'image_id': file_idx, 
                                          'image_path': os.path.join(img_dir, str(file_idx)+'.jpg'),
                                          'mask_path': os.path.join(mask_dir, str(file_idx)+'.png')}
        else:
            self.image_info = image_info
            
    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        img_path = self.image_info[idx]['image_path']
        image_id = int(img_path.split('/')[-1].split('.')[0])
        img = Image.open(img_path).convert('RGB')
        mask_path = self.image_info[idx]['mask_path']
        mask = Image.open(mask_path) 
        
        if self.img_transform:
            img = self.img_transform(img)
        
        mask_height = 1024 # mask.shape[1]
        mask_width = 1024  # mask.shape[2]

        mask = mask.resize((mask_width, mask_height), resample=Image.BILINEAR)        
                       
        ########## get bbox
        mask_cv2 = np.array(mask)    
        retval, bw = cv2.threshold(mask_cv2, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # find contour
        contours = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        n_objects = len(contours)
        masks = np.zeros((n_objects, mask_height, mask_width), dtype=np.uint8)        
        boxes = []
        for i, contour in enumerate(contours):
            x,y,w,h = cv2.boundingRect(contour)
            boxes.append([x, y, x+w, y+h])

            # fill contour
            crop_img = mask.crop(tuple(boxes[-1]))
            new_mask = Image.new(crop_img.mode, (mask_width, mask_height))
            new_mask.paste(crop_img, (x,y)) 
            a_mask = np.array(new_mask)
            
            a_mask = np.array(a_mask) > 0
            masks[i, :, :] = a_mask

        # dummy lables
        labels = [1 for _ in range(n_objects)]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([image_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((n_objects,), dtype=torch.int64)            

        target = {'boxes': boxes,
                  'labels': labels,
                  'masks': masks,
                  'image_id':  image_id,
                  'area': area,
                  'iscrowd': iscrowd}

        return img, target
