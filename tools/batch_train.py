import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import optim
from tqdm.notebook import tqdm
from model import ssl_maskrcnn
    
from tools.dataset import MaskRCNNDataset
from tools.functions import ssl_maskrcnn_train, ssl_maskrcnn_validation
from tools.visualize import save_masks


# define device, number of classes, transforms, and directories
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NORMALIZE = False
numOfClasses = 2

img_transform = transforms.Compose([
    transforms.Resize((14*64,14*64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mask_transform = transforms.Compose([
    transforms.Resize((32*32,32*32)),
    transforms.ToTensor(),
])


# test_dir: path of the challenge test data
# dinov2_weights: you can use pre-trained weights
# out_dir: output path of results. default = 'outputs'
def batch_train(train_dir, valid_dir, test_dir=None, out_dir='outputs',
                dino_models=['dinov2_b'],
                batchs=[16], epochs=100, dinov2_weights=None):

    if not os.path.exists(test_dir) or not os.path.isdir(test_dir):
        print(f'Error: {test_dir} does not exist.')
        return
        
    for m in dino_models:
        for b in batchs:
            # define variables
            base_name      = f'{m}_{b}batch_{epochs}epoch'
            base_dir       = os.path.join(out_dir, base_name)
            weights_dir    = os.path.join(base_dir, 'weights')
            pred_masks_dir = os.path.join(base_dir, 'pred_masks')
            final_weights  = os.path.join(base_dir, f'{base_name}_final_weights.pt')
            loss_plot_name = os.path.join(base_dir, f'{base_name}_loss_plot.png')

            print( '\n\n')
            print( '>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>> Current running: {base_dir}')
            print( '>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            
            # make folders
            os.makedirs(weights_dir, exist_ok=True)
            os.makedirs(pred_masks_dir, exist_ok=True)
    
            # define model
            model = ssl_maskrcnn(numOfClasses, backbone=m, dinov2_weights=dinov2_weights)
            model = model.to(device)
    
            # define optimizer and criterion
            optimizer = optim.Adam(model.parameters(), lr=5e-6)
            criterion = torch.nn.CrossEntropyLoss()
    
            train_loss_list = []
            val_loss_list = []

            train_image_dir = f'{train_dir}/image'
            train_mask_dir  = f'{train_dir}/mask'
            
            val_image_dir   = f'{valid_dir}/image'
            val_mask_dir    = f'{valid_dir}/mask'
            
            train_dataset = MaskRCNNDataset(img_dir=train_image_dir, mask_dir=train_mask_dir, num_classes = numOfClasses, img_transform=img_transform, mask_transform=mask_transform)
            valid_dataset = MaskRCNNDataset(img_dir=val_image_dir, mask_dir=val_mask_dir, num_classes = numOfClasses, img_transform=img_transform, mask_transform=mask_transform)
            
            train_loader = DataLoader(train_dataset, batch_size=b, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers=4)
            valid_loader = DataLoader(valid_dataset, batch_size=b, collate_fn=lambda x: tuple(zip(*x)), num_workers=4)

            # training
            with tqdm(total=epochs, desc='Train Progress', unit='epoch', position=0, leave=True) as pbar:
                for epoch in range(epochs):
                
                    train_loss = ssl_maskrcnn_train(model, train_loader, criterion, optimizer)
                    val_loss = ssl_maskrcnn_validation(model, valid_loader, criterion)
                    
                    train_loss_list.append(train_loss)
                    val_loss_list.append(val_loss)
                    
                    # save info every 20
                    if (epoch+1) % 20 == 0:
                        checkpoint_name = os.path.join(weights_dir, f'model_dfuc_ep_{epoch+1}.pt')
                        torch.save({'epoch': epoch+1, 
                                    'model_state_dict': model.state_dict(), 
                                    'optimizer_state_dict': optimizer.state_dict(), 
                                    'loss': train_loss, 
                                   }, checkpoint_name)
                   
                    pbar.set_postfix(train_loss=f'{train_loss:.4f}', valid_loss=f'{val_loss:.4f}')
                    pbar.update(1)
                    
            # save final weights
            torch.save(model.state_dict(), final_weights)
    
            # draw loss graph
            epoch_list = np.arange(1, epochs+1) # [1, 2, ..., epochs]
            plt.figure(figsize=(10, 5))
            plt.subplot(1,2,1)
            plt.xlabel('Epoch')
            plt.ylabel('Train_loss')
            plt.plot(epoch_list, train_loss_list)
            plt.subplot(1,2,2)
            plt.xlabel('Epoch')
            plt.ylabel('Val_loss')
            plt.plot(epoch_list, val_loss_list)
            
            plt.savefig(loss_plot_name)
            plt.show()

            # save pred masks
            save_masks(model, test_dir, pred_masks_dir, img_transform = img_transform)

            # rename folder to done
            os.rename(base_dir, base_dir+'_done')
