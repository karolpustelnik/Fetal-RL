import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torchvision.models import efficientnet_v2_l
#from .cbam import CBAMBlock

class SpatialAttention(torch.nn.Module):
    def __init__(self, feature_map_size = 16, n_channels = 1280):
        super().__init__()
    
        self.n_channels = n_channels
        self.feature_map_size = feature_map_size
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
        
        return attended_features
    
    
    
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
        attended_features = attended_features.permute(0, 2, 1)[:, :org_seq_len, :]
        attended_features = attended_features.mean(dim = 1)
        return attended_features


    

class EffnetV2_Key_Frame(torch.nn.Module):
    def __init__(self, out_features = 7, in_channels = 1, dropout = 0.4, use_sigmoid = False, 
                 use_attention = False, use_key_frame_attention = True,
                 max_len = 8):
        super().__init__()
        
        self.max_len = max_len
        self.use_key_frame_attention = use_key_frame_attention
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
        if self.use_attention:
            self.spatial_attention = SpatialAttention(feature_map_size = 16, n_channels=1280)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        if self.use_key_frame_attention:
            self.key_frame_attention = KeyFrameAttention(n_frames = 4, n_channels = 1280)
        
    def features_padding(self, features, max_length, org_seq_len):
        padding_length = max_length - org_seq_len
        padded_seq = torch.nn.functional.pad(features, (0, 0, 0, padding_length, 0, 0), mode='constant', value=0)
        return padded_seq
    
    def mask_sequence(self, padded_seq, org_seq_len):  

        mask = torch.zeros(padded_seq.size()[:2], dtype=torch.float32)
        # Set the non-padding elements to 1's
        mask[:, :org_seq_len] = 1

        # Expand the mask tensor to shape (1, 8, 1) and (1, 1, 8)
        mask_1 = mask.unsqueeze(1)
        mask_2 = mask.unsqueeze(2)

        # Compute the element-wise multiplication of the two mask tensors
        attention_mask = mask_1 * mask_2# (1, 8, 8)
        return attention_mask
        
    def count_params(self):
        
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def forward(self, x):
        org_seq_len = x.shape[0]
        features = self.model.features(x)
        if self.use_attention:
            features = self.spatial_attention(features)
        features = self.avgpool(features).squeeze(-1).permute(2, 0, 1)
        features_padded = self.features_padding(features, self.max_len, org_seq_len)
        mask = self.mask_sequence(features_padded, org_seq_len = x.shape[1])
        features_padded = features_padded.permute(0, 2, 1)
        if self.use_key_frame_attention:
            x = self.key_frame_attention(features_padded, org_seq_len = x.shape[1], Mask = mask)
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
    
    
model = EffnetV2_Key_Frame(out_features = 7, in_channels = 1, dropout = 0.4)
test_tensor = torch.rand(3, 1, 512, 512)

print(model(test_tensor))