import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torchvision.models import efficientnet_v2_l


class KeyFrameAttention(torch.nn.Module):
    def __init__(self, n_frames = 4, n_channels = 1280):
        super().__init__()
    
    
        self.n_frames = n_frames
        self.n_channels = n_channels
        self.keys = torch.nn.Conv1d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0)
        self.queries = torch.nn.Conv1d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0)
        self.values = torch.nn.Conv1d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0)
        self.refine = torch.nn.Conv1d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0)
        self.softmax = torch.nn.Softmax2d()
        self.alpha = torch.nn.Parameter(torch.zeros(1))
        
    def forward(self, x, org_seq_len, Mask = None):
        keys = self.keys(x) # (batch_size, n_frames, n_channels)
        queries = self.queries(x) # (batch_size, n_frames, n_channels)
        values = self.values(x) # (batch_size, n_frames, n_channels)
        matmul = torch.matmul(queries.permute(0, 2, 1), keys) # (batch_size, n_channels, n_frames)
        if Mask is not None:
            matmul = matmul.masked_fill(Mask == 0, -1e9)
        softmax = self.softmax(matmul) # (batch_size, n_channels, n_frames)
        attention_map = torch.matmul(values, softmax) # (batch_size, n_channels, n_frames)
        attended_features = self.refine(attention_map) # (batch_size, n_frames, n_channels)
        attended_features = self.alpha * attended_features + x
        print('attended features shape', attended_features.shape)
        attended_features = attended_features.permute(0, 2, 1)[:, :org_seq_len, :]
        print('attended features shape', attended_features.shape)
        attended_features = attended_features.mean(dim = 1)
        return attended_features
        
        
    
    
class ChannelAttention(torch.nn.Module):
    def __init__(self, feature_map_size = 16, n_channels = 1280):
        super().__init__()
    
        self.feature_map_size = feature_map_size
        self.n_channels = n_channels
        self.keys = torch.nn.Conv2d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0)
        self.queries = torch.nn.Conv2d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0)
        self.values = torch.nn.Conv2d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0)
        self.refine = torch.nn.Conv2d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0)
        self.softmax = torch.nn.Softmax2d()
        self.alpha = torch.nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        attended_features = torch.matmul(self.softmax(torch.matmul(self.keys(x).view(x.size(0), self.n_channels, -1), 
                                                                   self.queries(x).view(x.size(0), self.n_channels, -1).permute(0, 2, 1))), 
                                         self.values(x).view(x.size(0), self.n_channels, -1)) # (batch_size, n_channels, feature_map_size * feature_map_size)
        attended_features = attended_features.view(x.size(0), self.n_channels, self.feature_map_size, self.feature_map_size) # (batch_size, n_channels, feature_map_size, feature_map_size)
        attended_features = self.refine(attended_features)
        attended_features = self.alpha * attended_features + x
        
        return attended_features
    
    
class SpatialAttention(torch.nn.Module):
    def __init__(self, feature_map_size = 16, n_channels = 1280):
        super().__init__()
    
        self.feature_map_size = feature_map_size
        self.n_channels = n_channels
        self.keys = torch.nn.Conv2d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0)
        self.queries = torch.nn.Conv2d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0)
        self.values = torch.nn.Conv2d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0)
        self.refine = torch.nn.Conv2d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0)
        self.softmax = torch.nn.Softmax2d()
        self.alpha = torch.nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        
        attended_features = torch.matmul(self.softmax(torch.matmul(self.keys(x).view(x.size(0), self.n_channels, -1).permute(0, 2, 1), 
                                                                   self.queries(x).view(x.size(0), self.n_channels, -1))), 
                                         self.values(x).view(x.size(0), self.n_channels, -1).permute(0, 2, 1)) # (batch_size, feature_map_size * feature_map_size, n_channels)
        attended_features = attended_features.permute(0, 2, 1).view(x.size(0), self.n_channels, self.feature_map_size, self.feature_map_size) # (batch_size, n_channels, feature_map_size, feature_map_size)
        attended_features = self.refine(attended_features)
        attended_features = self.alpha * attended_features + x
        print(attended_features.shape)
        
        return attended_features

    
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
        self.spatial_attention = ChannelAttention(feature_map_size=16, n_channels=1280)
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
    
    
    
class EffnetV2(torch.nn.Module):
    def __init__(self, out_features = 7, in_channels = 1, dropout = 0.4, use_sigmoid = False):
        super().__init__()
        
        self.use_sigmoid = use_sigmoid
        self.dropout = dropout
        self.out_features = out_features
        self.in_channels = in_channels
        self.model = efficientnet_v2_l(weights = 'EfficientNet_V2_L_Weights.IMAGENET1K_V1')
        self.model.features[0] = torch.nn.Conv2d(self.in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier = torch.nn.Sequential(nn.Dropout(self.dropout), nn.Linear(1280, self.out_features))
        self.sigmoid = torch.nn.Sigmoid()
        
        
        
    def count_params(self):
        
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        print(x.shape)
        # x = x.view(x.size(0), 1280, -1)
        # x = self.model.avgpool(x)
        # x = x.squeeze(-1)
        # x = self.model.classifier(x)
        # x = self.model(x)
        # if self.use_sigmoid:
        #     x = self.sigmoid(x)
        return x
    
    
    
test_tesnor = (torch.rand(1, 1280, 3))

key_frame_attention = KeyFrameAttention(n_frames=8)

print(key_frame_attention(test_tesnor, 3).shape)