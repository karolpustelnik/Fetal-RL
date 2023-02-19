# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

from .fetal_loader import Fetal_frame, Fetal_vid_old, Fetal_vid_new, Fetal_frame_eval_reg, Fetal_frame_eval_cls

try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)


    sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=False
        )

    if config.PARALLEL_TYPE == 'model_parallel':
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, 
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            shuffle = True,
            drop_last = False)
        
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, 
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            shuffle = False,
            drop_last = False)
        
    elif config.PARALLEL_TYPE == 'ddp':
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, 
            sampler=sampler_train,
            shuffle = False,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=2,
            drop_last = True)
        
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, 
            sampler = sampler_val,
            shuffle = False,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=2,
            drop_last=False)


    # setup mixup / cutmix
    mixup_fn = None

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    affix = config.MODEL.AFFIX
    transform = build_transform(is_train, config, config.DATA.DATASET)
    if config.DATA.DATASET == 'fetal':
        if is_train:
            ann_path =  config.DATA.TRAIN_SET
            videos_path = config.DATA.VIDEOS_TRAIN
            #videos_path = f'/data/kpusteln/Fetal-RL/data_preparation/data_biometry/videos_train_{part}.csv'

        else:
            ann_path =  config.DATA.VAL_SET
            videos_path = config.DATA.VIDEOS_VAL
            #videos_path = f'/data/kpusteln/Fetal-RL/data_preparation/data_biometry/videos_val_{part}.csv'
    if config.PARALLEL_TYPE == 'model_parallel':
        dataset = Fetal_frame(root = config.DATA.DATA_PATH, ann_path = ann_path, transform = transform, img_scaling = config.DATA.IMG_SCALING)
        nb_classes = config.MODEL.NUM_CLASSES
        
    elif config.PARALLEL_TYPE == 'ddp':
            dataset = Fetal_frame(root = config.DATA.DATA_PATH, ann_path = ann_path, transform = transform, img_scaling = config.DATA.IMG_SCALING)
            nb_classes = config.MODEL.NUM_CLASSES
    if config.TRAIN.AUTO_RESUME == False:
        if config.MODEL.TASK_TYPE == 'reg':
            dataset = Fetal_frame_eval_reg(root = config.DATA.DATA_PATH, ann_path = ann_path, transform = transform)
            nb_classes = config.MODEL.NUM_CLASSES
        elif config.MODEL.TASK_TYPE == 'cls':
            dataset = Fetal_frame_eval_cls(root = config.DATA.DATA_PATH, ann_path = ann_path, transform = transform)
            nb_classes = config.MODEL.NUM_CLASSES



    return dataset, nb_classes


def build_transform(is_train, config, dataset_name):
    if dataset_name == 'fetal':
        if is_train:
            if config.DATA.AUGM:
                            t = transforms.Compose([
                            transforms.RandomRotation(degrees=(0, 5)),
                            transforms.RandomVerticalFlip(p=0.5),
                            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
                            transforms.RandomAutocontrast(p=0.5),])
            else:
                t = None
                #transforms.Normalize(mean=0.1354949, std=0.18222201)
        else:
            t = None

    return t