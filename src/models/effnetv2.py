import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torchvision.models import efficientnet_v2_l
from .cbam import CBAMBlock

    
class SpatialAttention(torch.nn.Module):
    def __init__(self, n_channels = 512):
        super().__init__()
    
        self.n_channels = n_channels
        self.keys = torch.nn.Conv2d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0)
        self.queries = torch.nn.Conv2d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0)
        self.values = torch.nn.Conv2d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0)
        self.refine = torch.nn.Conv2d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0)
        self.softmax = torch.nn.Softmax2d()
        self.alpha = torch.nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        output = torch.matmul(self.softmax(torch.matmul(self.queries(x), self.keys(x))),
                                 self.values(x))
        output = self.refine(output)
        output = self.alpha * output + x
        
        return output


    
class EffnetV2_L(torch.nn.Module):
    def __init__(self, out_features = 7, in_channels = 1, dropout = 0.4, use_sigmoid = False, use_attention = False):
        super().__init__()
        
        self.use_sigmoid = use_sigmoid
        self.use_attention = use_attention
        self.dropout = dropout
        self.out_features = out_features
        self.in_channels = in_channels
        self.model = efficientnet_v2_l(weights = 'EfficientNet_V2_L_Weights.IMAGENET1K_V1')
        self.model.features[0] = torch.nn.Conv2d(self.in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.avgpool = torch.nn.Identity()
        self.model.classifier = torch.nn.Sequential(nn.Dropout(self.dropout), nn.Linear(1280, self.out_features))
        self.sigmoid = torch.nn.Sigmoid()
        self.spatial_attention = SpatialAttention(n_channels=1280)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        
    def count_params(self):
        
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def forward(self, x):
        if self.use_attention:
            x = self.model.features(x)
            x = self.spatial_attention(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)
        else:
            x = self.model.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)
        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x


class EffnetV2_L_cbam(torch.nn.Module):
    def __init__(self, out_features = 7, in_channels = 1, dropout = 0.4):
        super().__init__()
        
        self.dropout = dropout
        self.out_features = out_features
        self.in_channels = in_channels
        self.model = efficientnet_v2_l(weights = 'EfficientNet_V2_L_Weights.IMAGENET1K_V1')
        self.model.features[0] = torch.nn.Conv2d(self.in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier = torch.nn.Identity()
        self.cbam = CBAMBlock(channel=1280, reduction=64, kernel_size=16*16)
        self.model.classifier = torch.nn.Sequential(nn.Dropout(self.dropout), nn.Linear(1280, self.out_features))
        self.linear = torch.nn.Linear(16*16, 1)
        self.sigmoid = torch.nn.Sigmoid()
        #self.classifier = torch.nn.Sequential(nn.Dropout(0.4), nn.Linear(1280, self.out_features))
        
        
        
    def count_params(self):
        
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def forward(self, x):
        x = self.model.features(x)
        x = self.cbam(x)
        #print(x.shape)
        x = x.view(x.size(0), 1280, -1)
        #print(x.shape)
        x = self.linear(x)
        x = x.squeeze(-1)
        #print(x.shape)
        x = self.model.classifier(x)
        x = self.sigmoid(x)
        return x


class EffnetV2_L_pos_encoding(torch.nn.Module):
    def __init__(self, out_features = 7, in_channels = 1):
        super().__init__()
        
        
        self.out_features = out_features
        self.in_channels = in_channels
        self.model = efficientnet_v2_l(weights = 'EfficientNet_V2_L_Weights.IMAGENET1K_V1')
        self.model.features[0] = torch.nn.Conv2d(self.in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        #self.model.classifier = torch.nn.Sequential(nn.Dropout(0.4), nn.Linear(1280, self.out_features))
        self.model.classifier = torch.nn.Identity()
        self.classifier = torch.nn.Sequential(nn.Dropout(0.4), nn.Linear(1280, self.out_features))
        
        
        # Positional embedding 
        max_len = 1000
        num_hiddens = 1280
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
        self.P = self.P.squeeze(0)
        
        
    def count_params(self):
        
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def forward(self, x):
        img = x[0]
        pos = x[1].cpu()
        features = self.model(img)
        
        positions = self.P[pos, :].cuda()
        out = features + positions
        out = self.classifier(out)
        return out, features
        




        
# test_tensor = torch.rand(1, 1, 448, 448).cuda()
# pos = torch.tensor(50).cuda()
# model = EffnetV2_L(out_features = 7, in_channels = 1).cuda()

# print(model((test_tensor, pos)).shape)


class EffnetV2_L_meta(torch.nn.Module):
    def __init__(self, out_features = 7, in_channels = 1, dropout = 0.4):
        super().__init__()
        
        
        self.dropout = dropout
        self.out_features = out_features
        self.in_channels = in_channels
        self.model = efficientnet_v2_l(weights = 'EfficientNet_V2_L_Weights.IMAGENET1K_V1')
        self.model.features[0] = torch.nn.Conv2d(self.in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier = torch.nn.Identity()
        #self.model.classifier = torch.nn.Sequential(nn.Dropout(0.4), nn.Linear(1280, self.out_features))
        self.classifier = torch.nn.Sequential(nn.Linear(1280 + 4, 512),
                                             nn.BatchNorm1d(512),
                                             torch.nn.SiLU(),
                                             nn.Dropout(self.dropout),
                                             torch.nn.Linear(512, out_features = self.out_features),) # 1280 + 64 meta feature (days, frame_location)
        self.meta = torch.nn.Sequential(nn.Linear(2, 4),
                                        nn.BatchNorm1d(4),
                                        nn.SiLU(),)
                                        
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        img = x[0]
        meta = x[1]
        meta = self.meta(meta)
        features = self.model(img)
        features = torch.cat([features, meta], dim = 1)
        out = self.classifier(features)
        return out
    
    
# model = EffnetV2_L()
# test_tensor = torch.rand(8, 1, 512, 512)

# print(model(test_tensor).shape)