    
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torchvision.models import efficientnet_v2_l, efficientnet_v2_s, efficientnet_v2_m
from .efficient_net_group_norm import effnetv2_m, effnetv2_l, effnetv2_xl
from .UniNet import UniNetB6
from .metaformer_baselines import CA_former
from .swin_transformer import SwinTransformer
#from .cbam import CBAMBlock

class SpatialAttention(torch.nn.Module):
    def __init__(self, feature_map_size = 16, n_channels = 1280, use_layer_norm = False, use_alpha = True, use_skip_connection = True, use_gelu = False):
        super().__init__()
    
        self.use_alpha = use_alpha
        self.use_skip_connection = use_skip_connection
        self.use_gelu = use_gelu
        self.use_layer_norm = use_layer_norm
        self.n_channels = n_channels
        self.feature_map_size = feature_map_size
        self.keys = torch.nn.Conv2d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.queries = torch.nn.Conv2d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.values = torch.nn.Conv2d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.refine = torch.nn.Conv2d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = torch.nn.Softmax2d()
        self.gelu = torch.nn.GELU()
        if self.use_alpha:
            self.alpha = torch.nn.Parameter(torch.zeros(1))
        if self.use_layer_norm:
            self.layer_norm = torch.nn.LayerNorm([self.n_channels, self.feature_map_size, self.feature_map_size])
    def forward(self, x):
     #   print('x in spatial attention', x.shape)
        attended_features = torch.matmul(self.softmax(torch.matmul(self.keys(x).view(x.size(0), self.n_channels, -1).permute(0, 2, 1), 
                                                                   self.queries(x).view(x.size(0), self.n_channels, -1))/self.n_channels**0.5), 
                                         self.values(x).view(x.size(0), self.n_channels, -1).permute(0, 2, 1)) # (batch_size, feature_map_size * feature_map_size, n_channels)
        attended_features = attended_features.permute(0, 2, 1).view(x.size(0), self.n_channels, self.feature_map_size, self.feature_map_size) # (batch_size, n_channels, feature_map_size, feature_map_size)
      #  print('attended_features', attended_features.shape)
        attended_features = self.refine(attended_features)
        if self.use_alpha:
            #print('spatial using alpha')
            attended_features = self.alpha*attended_features + x
        else:
            #print('spatial not using alpha')
            attended_features = attended_features + x
        if self.use_layer_norm:
            #print('spatial using layer norm')
            attended_features = self.layer_norm(attended_features)
        if self.use_gelu:
            #print('spatial using gelu')
            attended_features = self.gelu(attended_features)
        return attended_features
    
    
    
class KeyFrameAttention(torch.nn.Module):
    def __init__(self, n_frames = 4, n_channels = 1280, use_alpha = False, use_layer_norm = False, use_skip_connection = False, use_gelu = False):
        super().__init__()
    
        self.use_alpha = use_alpha
        self.use_skip_connection = use_skip_connection
        self.use_gelu = use_gelu
        self.use_layer_norm = use_layer_norm
        self.n_frames = n_frames
        self.n_channels = n_channels
        self.keys = torch.nn.Linear(self.n_channels, self.n_channels, bias=False)
        self.queries = torch.nn.Linear(self.n_channels, self.n_channels, bias=False)
        self.values = torch.nn.Linear(self.n_channels, self.n_channels, bias=False)
        self.refine = torch.nn.Linear(self.n_channels, self.n_channels, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        if self.use_layer_norm:
            self.layer_norm = torch.nn.LayerNorm([self.n_frames, self.n_channels])
        self.gelu = torch.nn.GELU()
        if self.use_alpha and self.use_skip_connection:
            self.alpha = torch.nn.Parameter(torch.zeros(1))
        
    def forward(self, x, Mask = None):
        # x shape: 
        #print('x shape', x.shape)
        keys = self.keys(x) # (batch_size, n_frames, n_channels)
        #print('keys', keys.shape)
        queries = self.queries(x) # (batch_size, n_frames, n_channels)
        #print('queries', queries.shape)
        values = self.values(x) # (batch_size, n_frames, n_channels)
        #print('values', values.shape)
        matmul = torch.matmul(queries, keys.permute(0, 2, 1)).float() # (batch_size, n_channels, n_frames)
        #print('matmul', matmul.shape)
        if Mask is not None:
            matmul = matmul.masked_fill(Mask == 0, -1e20)
        #print('matmul shape', matmul.shape)
        softmax = self.softmax(matmul/(self.n_channels) ** 0.5) # (batch_size, n_channels, n_frames)
        attention_map = torch.matmul(values.permute(0, 2, 1), softmax) # (batch_size, n_channels, n_frames)
        #print('attention_map', attention_map.shape)
        #print('attention_map', attention_map.shape)
        #print('attention_map', attention_map)
        attended_features = self.refine(attention_map.permute(0, 2, 1)) # (batch_size, n_frames, n_channels)
        if self.use_skip_connection:
            #print('kfa using skip connection')
            if self.use_alpha:
               # print('kfa using alpha')
                attended_features = self.alpha*attended_features + x
            else:
               # print('kfa not using alpha')
                attended_features = attended_features + x
        if self.use_layer_norm:
           # print('kfa using layer norm')
            attended_features = self.layer_norm(attended_features)
        if self.use_gelu:
           # print('kfa using gelu')
            attended_features = self.gelu(attended_features)
        #print('attended_features', attended_features)
        #attended_features = attended_features + x
        #print('attended_features', attended_features)
        #attended_features = attended_features[:, :org_seq_len, :]
        #attended_features = attended_features.permute(0, 2, 1)
        #print('attended_features', attended_features)
        #print('attended_features', attended_features.shape)
        #print('attended_features', attended_features)
        return attended_features




class EffnetV2_Key_Frame(torch.nn.Module):
    def __init__(self, out_features = 7, in_channels = 1, dropout = 0.4, use_sigmoid = False, 
                 use_attention = True, use_key_frame_attention = False,
                 n_frames = 4, use_alpha = True, use_layer_norm = False, use_skip_connection = False, use_gelu = False, use_head = False, backbone = 'effnetv2'):
        super().__init__()
        
        self.backbone = backbone
        self.use_head = use_head
        self.use_alpha = use_alpha
        self.use_layer_norm = use_layer_norm
        self.use_skip_connection = use_skip_connection
        self.use_gelu = use_gelu
        self.n_frames = n_frames
        self.use_key_frame_attention = use_key_frame_attention
        self.use_sigmoid = use_sigmoid 
        self.use_attention = use_attention
        self.dropout = dropout
        self.out_features = out_features
        self.in_channels = in_channels
        if self.backbone == 'effnetv2':
           # print('Using EffnetV2')
            self.model = efficientnet_v2_l(weights ='DEFAULT')
            self.model.features[0] = torch.nn.Conv2d(self.in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.avgpool = torch.nn.Identity()
            self.model.classifier = torch.nn.Sequential(nn.Dropout(self.dropout), nn.Linear(1280, self.out_features))
        elif self.backbone == 'uninet':
         #   print('Using UniNet')
            self.model = UniNetB6()
        elif self.backbone == 'caformer':
         #   print('Using CAFormer')
            self.model = CA_former()
        elif self.backbone == 'swin_transformer':
         #   print('Using Swin Transformer')
            self.model = SwinTransformer()
        self.sigmoid = torch.nn.Sigmoid()
        if self.use_head:
            self.head = torch.nn.Linear(self.n_frames, self.out_features)
        
        if self.use_attention:
           # print('Using Attention')
            self.spatial_attention = SpatialAttention(feature_map_size = 16, n_channels=1280,
                                                      use_alpha = self.use_alpha,
                                                      use_layer_norm = self.use_layer_norm,
                                                      use_skip_connection = self.use_skip_connection,
                                                      use_gelu = self.use_gelu)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        if self.use_key_frame_attention:
          #  print('Using Key Frame Attention')
            self.key_frame_attention = KeyFrameAttention(n_frames = self.n_frames, n_channels = 1280,
                                                         use_alpha = self.use_alpha,
                                                    use_layer_norm = self.use_layer_norm,
                                                      use_skip_connection = self.use_skip_connection,
                                                      use_gelu = self.use_gelu)
        
    def features_padding(self, features, max_length, split_sizes):
        padded_output = [torch.nn.functional.pad(feature, (0, max_length - split_sizes[i], 0, 0, 0, 0), mode='constant', value=0) for i, feature in enumerate(features)]
        padded_output = torch.stack(padded_output)
        return padded_output
    
    def mask_sequence(self, padded_seq, org_seq_lens):  
        
        # Define the padded sequence
        padded_seq_len = padded_seq.size(-1)
        batch_len = padded_seq.size(0)
        # Define the mask tensor
        mask = torch.zeros((batch_len, padded_seq_len, padded_seq_len), dtype=torch.float32) 
        
        # Set the non-padding elements to 1's
        for i in range(batch_len):
            mask[i, :, :org_seq_lens[i]] = 1
        
        
        return mask
        
    def count_params(self):
        
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    
    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward
    
    def forward(self, x, org_seq_len):

        #print('shape of x', x.shape)
        x = torch.cat(x)
      #  print('shape of before RFE x', x.shape)
        if self.backbone == 'effnetv2':
            features = self.model.features(x)
        elif self.backbone == 'uninet':
            features = self.model.forward_features(x).unsqueeze(0).permute(0, 2, 1)
        elif self.backbone == 'caformer' or self.backbone == 'swin_transformer':
            features = self.model.forward(x).unsqueeze(0).permute(0, 2, 1)
        elif self.backbone == 'swin_transformer':
            self.model.forward(x).unsqueeze(0).permute(0, 2, 1)
            
        #features = checkpoint_sequential(self.model.features, segments=len(self.model.features), input=x)
        if self.use_attention:
            features = self.spatial_attention(features)
        
        #print('features shape', features.shape)
        if self.backbone == 'effnetv2':
            features = self.avgpool(features).squeeze(-1).permute(2, 1, 0)
      #  print('features shape', features.shape)
        tensor_list = torch.split(features, split_size_or_sections = org_seq_len, dim=2)
        
        features_padded = self.features_padding(tensor_list, self.n_frames, org_seq_len)
        #print('features_padded shape', features_padded.shape)
        #print('features_padded shape', features_padded.shape)
        features_padded = features_padded.squeeze(1)
        #print('features_padded shape', features_padded.shape)
        mask = self.mask_sequence(features_padded, org_seq_lens = org_seq_len).cuda() if torch.cuda.is_available() else self.mask_sequence(features_padded, org_seq_lens = org_seq_len)
        #print('mask shape', mask.shape)
        
        if self.use_key_frame_attention:
          #  print('Using Key Frame Attention')
            x = self.key_frame_attention(features_padded.permute(0, 2, 1), Mask = mask)
            if self.backbone == 'caformer':
                x = self.model.model.head(x)
            elif self.backbone == 'swin_transformer':
                x = self.model.head(x)
            else:
                x = self.model.classifier(x)
        else:
          #  print('Not Using Key Frame Attention')
            #print(features_padded.shape)
            #features_padded = features_padded.mean(dim = 1)
            features_padded = features_padded.permute(0, 2, 1)
            if self.backbone == 'caformer':
                x = self.model.model.head(features_padded)
            elif self.backbone == 'swin_transformer':
                x = self.model.head(features_padded)
            else:
                x = self.model.classifier(features_padded)
            #print('x shape in else', x.shape)
            #print(x.shape)
        if self.use_sigmoid:
         #   print('Using Sigmoid')
            x = self.sigmoid(x)
        if self.use_head:
            x = x.permute(0, 2, 1).squeeze(1)
            x = self.head(x)
        else:
            x = torch.stack([batch_element[:org_seq_len[i], :].mean(dim=0) for i, batch_element in enumerate(x)])
        return x
    


        
    
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
        if self.use_attention:
            self.spatial_attention = SpatialAttention(feature_map_size = 16, n_channels=1280)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        
    def count_params(self):
        
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def forward(self, x):
        if self.use_attention:
            x = self.model.features(x)
            #x = self.model.conv(x)
            x = self.spatial_attention(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)
        else:
            x = self.model.features(x)
            #x = self.model.conv(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)
        if self.use_sigmoid:
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
        self.model = efficientnet_v2_s(weights = 'EfficientNet_V2_s_Weights.IMAGENET1K_V1')
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
    
# model = EffnetV2_L(out_features = 2, in_channels = 1, dropout = 0.2)
# test_tensor = torch.rand(1, 1, 512, 512)
# print(model.count_params())
# print(model(test_tensor).shape)

# model = EffnetV2_Key_Frame(out_features = 1, in_channels = 1, dropout = 0.4, 
#                            use_key_frame_attention=True, use_layer_norm = False,
#                            use_skip_connection = True, use_gelu = False, use_attention = True,
#                            backbone = 'effnetv2')
# split_sizes = [4, 2, 3, 4]
# org_batch = [torch.rand(i, 1, 512, 512) for i in split_sizes]

# print(model(org_batch, split_sizes).shape)

