# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = '/data/kpusteln/fetal/fetal_extracted/'
_C.DATA.PATH_PREFIX = '/data'
# train set path
_C.DATA.TRAIN_SET = '/data/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/biometry_scaled_ps/all/biometry_train_scaled_size_all.csv'
# val set path
_C.DATA.VAL_SET = '/data/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/biometry_scaled_ps/all/biometry_val_scaled_size_all.csv'
# Dataset name
_C.DATA.DATASET = 'imagenet'

_C.DATA.VIDEOS_TRAIN = 'videos_train'
_C.DATA.VIDEOS_VAL = 'videos_val'
_C.DATA.VIDEOS_TEST = 'videos_test'
_C.DATA.IMG_SCALING = True
# whether use augmentation

_C.DATA.AUGM = True
# Input image size
_C.DATA.IMG_SIZE = 512
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.PART = 'abdomen'

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
_C.MODEL.RUN_ID = 'default' # comet ml run id
_C.MODEL.PROJECT_NAME = 'default' #comet ml project name
# Model name
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
_C.MODEL.RESUME_DETECTOR = ''
_C.MODEL.RESUME_REGRESOR = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 7
# Dropout rate
_C.MODEL.DROP_RATE = 0.0

_C.MODEL.KEY_FRAME_ATTENTION = True

_C.MODEL.SIGMOID = False
_C.MODEL.ATTENTION = False
# Task type
_C.MODEL.TASK_TYPE = 'reg'
_C.MODEL.AFFIX = '.csv'
_C.MODEL.BODY_PART = 'abdomen'
# Drop path rate
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1
_C.MODEL.LOSS = 'L1'



# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
_C.TRAIN.NUM_FRAMES = 4
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 3
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = ''
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Misc
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
# for acceleration
_C.PARALLEL_TYPE = 'model_parallel'

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.train_path:
        config.DATA.TRAIN_SET = args.train_path
    if args.val_path:
        config.DATA.VAL_SET = args.val_path
    if args.zip:
        config.DATA.ZIP_MODE = True
    if args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.path_prefix:
        config.DATA.PATH_PREFIX = args.path_prefix
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.amp_opt_level:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
        if args.amp_opt_level == 'O0':
            config.AMP_ENABLE = False
    if args.disable_amp:
        config.AMP_ENABLE = False
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True
    if args.augm:
        config.DATA.AUGM = True if args.augm == 'True' else False
    if args.dropout:
        config.MODEL.DROP_RATE = args.dropout
    if args.base_lr:
        config.TRAIN.BASE_LR = args.base_lr
    if args.optimizer:
        config.TRAIN.OPTIMIZER.NAME = args.optimizer
    if args.scaling:
        config.DATA.IMG_SCALING = True if args.scaling == 'True' else False
    if args.sigmoid:
        config.MODEL.SIGMOID = True if args.sigmoid == 'True' else False
    if args.loss:
        config.MODEL.LOSS = args.loss
    if args.attention:
        config.MODEL.ATTENTION = True if args.attention == 'True' else False
    if args.key_frame_attention:
        config.MODEL.KEY_FRAME_ATTENTION = True if args.key_frame_attention == 'True' else False
    if args.num_frames:
        config.TRAIN.NUM_FRAMES = args.num_frames

    

    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
