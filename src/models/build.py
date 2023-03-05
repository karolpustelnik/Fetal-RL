    
import warnings
warnings.filterwarnings("ignore")
from .effnet_longformer import EFL
from .UniNet import UniNetB6
from .vit import ViT
from .effnetv2 import EffnetV2_L_meta, EffnetV2_Key_Frame, EffnetV2_L
from .efficient_ete import EffnetEtE
from .metaformer_baselines import CA_former
from .D_models.swin_video import SwinTransformer3D
    
def build_model(config):
    model_type = config.MODEL.TYPE
    print('Model type: ', model_type)
    print('using bacbkone: ', config.MODEL.BACKBONE)

    if model_type == 'effnetv2' or model_type == 'effnetv2_cls_pos_encoding':
        model = EffnetV2_L(out_features = config.MODEL.NUM_CLASSES, dropout= config.MODEL.DROP_RATE, use_sigmoid = config.MODEL.SIGMOID, use_attention = config.MODEL.ATTENTION)
    elif model_type == 'swin-video':
        model = SwinTransformer3D()
    elif model_type == 'efl':
        model = EFL(out_features = config.MODEL.NUM_CLASSES)
    elif model_type == 'uninet':
        model = UniNetB6(out_features = config.MODEL.NUM_CLASSES)
    elif model_type == 'vit':
         model = ViT(512, 32, 1, 1024, 6, 16, 2048, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.)
    elif model_type == 'ete':
        model = EffnetEtE(out_features = config.MODEL.NUM_CLASSES, in_channels = 1)
    elif model_type == 'two_models':
        detector = EffnetV2_L(out_features = config.MODEL.NUM_CLASSES)
        regresor = EffnetV2_L(out_features = 1)
        model = (detector, regresor)
    elif model_type == 'effnetv2_meta':
        model = EffnetV2_L_meta(out_features = config.MODEL.NUM_CLASSES, dropout= config.MODEL.DROP_RATE)
    elif model_type == 'caformer':
        model = CA_former(out_features = config.MODEL.NUM_CLASSES)
    elif model_type == 'effnetv2_key_frame':
        model = EffnetV2_Key_Frame(out_features = config.MODEL.NUM_CLASSES, dropout= config.MODEL.DROP_RATE, 
                                   use_sigmoid = config.MODEL.SIGMOID, use_attention = config.MODEL.ATTENTION,
                                   use_key_frame_attention=config.MODEL.KEY_FRAME_ATTENTION, n_frames=config.TRAIN.NUM_FRAMES,
                                   use_alpha = config.MODEL.USE_ALPHA, use_layer_norm=config.MODEL.USE_BATCH_NORM,
                                   use_skip_connection=config.MODEL.USE_SKIP_CONNECTION, use_gelu = config.MODEL.USE_GELU,
                                   use_head = config.MODEL.USE_HEAD, backbone = config.MODEL.BACKBONE)
        
        
    return model
