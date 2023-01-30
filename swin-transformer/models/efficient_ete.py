# This script creates a model for fetal videos in a end to end manner
# Backbone for classification is EfficienNet V2
# Bacbkone for regression is EfficientNet V2

# EffnetEtE - Efficient End to End model for fetal videos



import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torchvision.models import efficientnet_v2_s





class EffnetEtE(torch.nn.Module):
    def __init__(self, out_features = 7, in_channels = 1):
        super(EffnetEtE, self).__init__()
        
        
        self.out_features = out_features
        self.in_channels = in_channels
        self.detector = efficientnet_v2_s(weights = 'EfficientNet_V2_S_Weights.IMAGENET1K_V1').to('cuda:0', dtype=torch.float32)
        self.detector.features[0] = torch.nn.Conv2d(self.in_channels, 24, kernel_size=(3, 3), 
                                                    stride=(2, 2), padding=(1, 1), bias=False).to('cuda:0', dtype=torch.float32)
        self.detector.classifier = torch.nn.Sequential(nn.Linear(1280, self.out_features)).to('cuda:0', dtype=torch.float32)
        self.regressor = efficientnet_v2_s(weights = 'EfficientNet_V2_S_Weights.IMAGENET1K_V1').to('cuda:1', dtype=torch.float32)
        self.regressor.features[0] = torch.nn.Conv2d(self.in_channels, 24, kernel_size=(3, 3), 
                                                     stride=(2, 2), padding=(1, 1), bias=False).to('cuda:1', dtype=torch.float32)
        
        self.regressor.classifier = torch.nn.Sequential(nn.Linear(1280, 1)).to('cuda:1', dtype=torch.float32)
        
        
        
    def count_params(self):
        
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def forward(self, x):
        frames = x[0]
        x, position_ids = x #x: B - batch size, C - channels, F - frames, H - height, W - widt
        x = x.to('cuda:0', dtype=torch.float32)
        position_ids = position_ids.to('cuda:1', dtype=torch.float32)
        # spatial backbone
        B, C, F, H, W = x.shape # B - batch size, C - channels, F - frames, H - height, W - width
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B * F, C, H, W)
        #print(x.dtype)
        chunk_size = 8
        #n_chunks = int(B*F/chunk_size) + 1
        #print(n_chunks)
        x = torch.split(x, chunk_size, dim=0)
        backbone_output = []
        for i, chunk in enumerate(x):
            #print(i)
            chunk = self.detector(chunk)
            #print(torch.cuda.memory_allocated(0))
            #print(chunk.shape)
            chunk = chunk.to('cuda:1', dtype=torch.float32)
            #print(torch.cuda.memory_allocated(0))
            chunk = chunk.unsqueeze(0)
            backbone_output.append(chunk)
            
        detector_output = torch.cat(backbone_output, dim=1) # [B, F, out_features]
        # sum logits to know which video is loaded
        logits_sum = detector_output.sum(dim=1) # [B, out_features]
        print(f'shape of logits_sum:{logits_sum.shape}')
        types_sorted = torch.argsort(logits_sum, dim=1, descending=True) # [B, out_features]
        types_sorted = types_sorted.squeeze(0)
        """labels:
        0 - other
        1 - head non-standard plane
        2 - head standard plane
        3 - abdomen non-standard plane
        4 - abdomen standard plane
        5 - femur non standard plane
        6 - femur standard plane
        """
        print(f'types_sorted:{types_sorted}')
        if types_sorted[0] != 0:
            video_type = types_sorted[0]
        else:
            video_type = types_sorted[1]
            
        if video_type == 1 or video_type == 2:
            x = detector_output[:, :, 2:3]
        elif video_type == 3 or video_type == 4:
            x = detector_output[:, :, 4:5]
        elif video_type == 5 or video_type == 6:
            x = detector_output[:, :, 6:7]
        
        print(x.shape)
        x = x.permute(0, 2, 1) # [B, F, C]
        
        logits_sorted = torch.argsort(x, dim=2, descending=True) # [B, C, F]
        logits_sorted = logits_sorted.squeeze(0).squeeze(0) # [F]
        #take 32 most important frames
        top_frames = logits_sorted[:32]  # [32]
        selected_frames = frames[:, :, top_frames, :, :]
        
        
        
        
        ########## REGRESSION ##########
        selected_frames = selected_frames.permute(0, 2, 1, 3, 4)
        selected_frames = selected_frames.squeeze(0)
        selected_frames = selected_frames.to('cuda:1', dtype=torch.float32)
        print(f'shape of selected_frames:{selected_frames.shape}')
        regr_output = self.regressor(selected_frames)
        #take mean of all frames
        regr_output = regr_output.mean()
        regr_output = regr_output.reshape(1,1)
        print(f'shape of regr_output:{regr_output}')
        
            
            
            
        
        
        detector_output = detector_output.squeeze(0)
        return detector_output, regr_output
    
    
# test_tensor = torch.rand(1, 1, 180, 224, 224)

# positions = torch.tensor([[i for i in range(240)]])

# model = EffnetEtE(out_features = 7, in_channels = 1)

# print(model((test_tensor, positions))[1].shape)






