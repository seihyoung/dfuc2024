import torch
import torch.nn as nn
from torch.hub import load
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights


RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)
box_detections_per_img = 10


dino_backbones = {
    'dinov2_s':{
        'name':'dinov2_vits14',
        'embedding_size':384,
        'patch_size':14
    },
    'dinov2_b':{
        'name':'dinov2_vitb14',
        'embedding_size':768,
        'patch_size':14
    },
    'dinov2_l':{
        'name':'dinov2_vitl14',
        'embedding_size':1024,
        'patch_size':14
    },
    'dinov2_g':{
        'name':'dinov2_vitg14',
        'embedding_size':1536,
        'patch_size':14
    },
}

def get_dino_models():
    print('dino_model : {name, embedding_size, patch_size}')
    print('-----------------------------------------------')
    for k, v in dino_backbones.items():
        print(f'{k}: {v}')
        
class ssl_maskrcnn_head(nn.Module):
    def get_model(self, normalize=True):
        self.normalize = normalize 

        # common parameters
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        args = {
            'weights': weights,
            'box_detections_per_img': box_detections_per_img
        }
        
        # normalize if needed
        if self.normalize:
            args['image_mean'] = RESNET_MEAN
            args['image_std'] = RESNET_STD
        
        # 모델 생성
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(**args)
        
        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        # get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # replace mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, self.num_classes)
        
        return model

    def __init__(self, embedding_size = 384, num_classes = 5, normalize=True):    
        super(ssl_maskrcnn_head, self).__init__()
        self.num_classes = num_classes
        self.adjustment_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embedding_size, 64, (3,3), padding=(1,1)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, (3,3), padding=(1,1)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, (3,3), padding=(1,1)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 3, (3,3), padding=(1,1)),            
        )
        self.maskrcnn_model = self.get_model(normalize=True)

    def forward(self, x, y):
        x = self.adjustment_conv(x)     
        x = self.maskrcnn_model(x, y)
        return x

    
class ssl_maskrcnn(nn.Module):
    def __init__(self, num_classes, backbone = 'dinov2_s', head = 'maskrcnn', backbones = dino_backbones, normalize=True, dinov2_weights=None):
        super(ssl_maskrcnn, self).__init__()
        self.heads = {
            'maskrcnn': ssl_maskrcnn_head
        }
        self.backbones = dino_backbones

        self.backbone = load('facebookresearch/dinov2', self.backbones[backbone]['name'])

        if dinov2_weights:
            self.backbone.load_state_dict(torch.load(dinov2_weights))
            print(f'Weights data {dinov2_weights} is loaded......')
        
        self.backbone.eval()
        self.num_classes =  num_classes # add a class for background if needed
        self.normalize = normalize
        self.embedding_size = self.backbones[backbone]['embedding_size']
        self.patch_size = self.backbones[backbone]['patch_size']
        self.head = self.heads[head](self.embedding_size,self.num_classes, self.normalize)

    def forward(self, x, y):
        x = torch.stack(list(x), dim=0)
        batch_size = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size) 
        with torch.no_grad():
            x = self.backbone.forward_features(x.cuda())
            x = x['x_norm_patchtokens']
            x = x.permute(0,2,1)
            x = x.reshape(batch_size,self.embedding_size,int(mask_dim[0]),int(mask_dim[1]))
        x = self.head(x, y)
        
        return x