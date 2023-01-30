import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EffNet(torch.nn.Module):
    def __init__(self, out_features = 7, use_pretrained = True, extract = True, freeze = True, unfreeze_last_layers = True):
        super(EffNet, self).__init__()
        self.out_features = out_features
        self.extract = extract
        self.backbone = EfficientNet.from_pretrained('efficientnet-b6', in_channels = 1, num_classes=self.out_features)
        if use_pretrained:
            model = torch.load('/data/kpusteln/Fetal-RL/swin-transformer/output/effnet_reg_v2_abdomen/default/ckpt_epoch_61.pth')['model']
            for key in list(model.keys()):
                if 'backbone' in key:
                    model[key.replace('backbone.', '')] = model.pop(key) # remove prefix backbone.
            self.backbone.load_state_dict(model)
        if self.extract:    ## extract features for the transformer, ignore last layer
            self.backbone._fc = torch.nn.Identity()
        if freeze:
            for param in self.backbone.parameters():
                    param.requires_grad = False
                
        if unfreeze_last_layers:
            for param in self.backbone._blocks[44:].parameters():
                    param.requires_grad = True
                
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.backbone(x)
        
        return x
    


class RegFormer(torch.nn.Module):
    def __init__(self, out_features = 1, use_pretrained = True, extract = True, freeze = True, unfreeze_last_layers = True, task_type = 'reg'):
        super(RegFormer, self).__init__()
        self.out_features = out_features
        self.extract = extract
        self.task_type = task_type
        self.backbone = EfficientNet.from_pretrained('efficientnet-b6', in_channels = 1, num_classes=self.out_features)
        if use_pretrained:
            model = torch.load('/data/kpusteln/Fetal-RL/swin-transformer/output/effnet_reg_v2_abdomen/default/ckpt_epoch_61.pth')['model']
            for key in list(model.keys()):
                if 'backbone' in key:
                    model[key.replace('backbone.', '')] = model.pop(key) # remove prefix backbone.
            self.backbone.load_state_dict(model)
        if self.extract:    ## extract features for the transformer, ignore last layer
            self.backbone._fc = torch.nn.Identity()
        if freeze:
            for param in self.backbone.parameters():
                    param.requires_grad = False
                
        if unfreeze_last_layers:
            for param in self.backbone._blocks[44:].parameters():
                    param.requires_grad = True
        self.feature_dim = 2304
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.hidden_size = 512
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 1 if self.task_type == 'reg' else self.out_features))
        
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.backbone(x)
        x = self.transformer_encoder(x)
        x = self.mlp_head(x)
        return x


regformer = RegFormer(use_pretrained=False, freeze=False, unfreeze_last_layers=True, task_type='reg')
regformer = regformer.cuda()
print(regformer.count_params())
test_tensor = torch.rand(8, 1, 512, 512)
test_tensor = test_tensor.cuda()
print(regformer(test_tensor).shape)
