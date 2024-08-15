import os
from PIL import Image
import torch
import numpy as np
from tqdm.notebook import tqdm
from tools.score import dice


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ssl_maskrcnn_train(model, train_loader, criterion, optimizer):
    loss_accum = 0.
    samples = 0.
    ret_loss = 0.

    model.train()
    loop = tqdm(train_loader, desc='Train progress', total=len(train_loader), position=1, leave=False)
    for images, targets in loop:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        samples += len(images)
        
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # logging
        loss_accum += loss.item()

        # Update tqdm progress bar with the loss
        ret_loss = loss_accum/samples
        loop.set_postfix(train_loss=f'{ret_loss:.4f}')
        
    return ret_loss


def ssl_maskrcnn_validation(model, valid_loader, criterion):    
    loss_accum = 0.
    samples = 0.
    ret_loss = 0.

    model.eval()
    with torch.no_grad():
        loop = tqdm(valid_loader, desc='Valid progress', total=len(valid_loader), position=2, leave=False)
        for images, targets in loop:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            output = model(images, targets)
            
            samples_num = len(images)
            samples += samples_num
                        
            for t in range(samples_num):
                targets_or = np.zeros((1024, 1024), dtype=bool)
                for n in range(len(targets[t]['masks'])):
                    targets_or = np.logical_or(targets_or, targets[t]['masks'][n].to('cpu').reshape(1024, 1024) > 0.6)

                score_num = len(output[t]['scores'])

                preds_or = np.zeros((1024, 1024), dtype=bool)
                for n in range(score_num):
                    preds_or = np.logical_or(preds_or, output[t]['masks'][n].to('cpu').reshape(1024, 1024) > 0.6)
                    
                loss_accum += dice(targets_or, preds_or)
                    
            # Update tqdm progress bar with the loss
            ret_loss = 1. - loss_accum/samples
            loop.set_postfix(valid_loss=f'{ret_loss:.4f}')
        
    return ret_loss


def ssl_maskrcnn_infer(model, device, image_path=None, img_transform=None, image=None):
    if image == None:
        # Load and transform the image
        image = Image.open(image_path).convert('RGB')
        transformed_image = img_transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    else:
        transformed_image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Make sure the model is in evaluation mode
    model.eval()

    with torch.no_grad():
        # Make prediction
        preds = model(transformed_image, None)[0]

    return transformed_image.cpu().squeeze().permute(1, 2, 0).numpy(), preds
