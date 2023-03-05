# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
    
import warnings
warnings.filterwarnings("ignore")

import dill
import dill as pickle
import os
import torch
import numpy as np
import torch.distributed as dist
import torchvision
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from .fetal_loader import Fetal_frame, Fetal_frame_eval_reg, Fetal_frame_eval_cls, Video_Loader, Eval_Video_Loader

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

class AdjustContrast(object):
    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def __call__(self, img):
        return torchvision.transforms.functional.adjust_contrast(img, self.contrast_factor)
    
class MyCollate:
    def __init__(self):
        pass
        
    def __call__(self, batch):
        images = [item[0] for item in batch]
        Classes = [item[1] for item in batch]
        measure = torch.stack([item[2] for item in batch])
        ps = torch.stack([item[3] for item in batch])
        frame_n = [item[4] for item in batch]
        measure_scaled = torch.stack([item[5] for item in batch])
        index = [item[6] for item in batch]
        days_normalized = [item[7] for item in batch]
        frame_loc = [item[8] for item in batch]
        measure_normalized = torch.stack([item[9] for item in batch])
        org_seq_lens = [item[10] for item in batch]
        images_lens = [len(img) for img in images]
        assert org_seq_lens == images_lens, 'images are not equal to org_seq_lens'
        return images, Classes, measure, ps, frame_n, measure_scaled, index, days_normalized, frame_loc, measure_normalized, org_seq_lens
    
class MyCollate_Reg:
    def __init__(self):
        pass
        
    def __call__(self, batch):
        images = [item[0] for item in batch]
        Classes = [item[1] for item in batch]
        videos = [item[2] for item in batch]
        ps = torch.stack([item[3] for item in batch])
        lens = [item[4] for item in batch]
        images_lens = [len(img) for img in images]
        assert lens == images_lens, 'images are not equal to lens'
        return images, Classes, videos, ps, lens
    
    
    
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

    collater = MyCollate_Reg() if config.EVAL_MODE else MyCollate()
        
    if config.PARALLEL_TYPE == 'model_parallel':
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, 
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=8,
            shuffle = True,
            drop_last = False)
        
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, 
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=8,
            shuffle = False,
            drop_last = False)
        
    elif config.MODEL.TASK_TYPE == 'reg':
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, 
            sampler=sampler_train,
            shuffle = False,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=4,
            drop_last = True,
            collate_fn = collater)
        
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, 
            sampler = sampler_val,
            shuffle = False,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=4,
            drop_last=True if config.TRAIN.AUTO_RESUME else False,
            collate_fn = collater)
        
    elif config.MODEL.TASK_TYPE == 'cls':
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, 
            sampler=sampler_train,
            shuffle = False,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=8,
            drop_last = True,)
        
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, 
            sampler = sampler_val,
            shuffle = False,
            batch_size=1,
            num_workers=8,
            drop_last=True if config.TRAIN.AUTO_RESUME else False,)
    


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
        print('model parallel: Fetal_frame')
        dataset = Fetal_frame(root = config.DATA.DATA_PATH, ann_path = ann_path, transform = transform, img_scaling = config.DATA.IMG_SCALING)
        nb_classes = config.MODEL.NUM_CLASSES
        
    elif config.PARALLEL_TYPE == 'ddp':
        
        if config.MODEL.TYPE == 'effnetv2_key_frame' or config.MODEL.TYPE == 'swin-video':
            print(config.MODEL.TYPE)
            print('ddp: Video_Loader')
            dataset = Video_Loader(root = config.DATA.DATA_PATH, videos_path = videos_path,ann_path = ann_path, 
                                   transform = transform, img_scaling = config.DATA.IMG_SCALING, num_frames = config.TRAIN.NUM_FRAMES, img_size = config.DATA.IMG_SIZE)
        elif config.MODEL.TYPE == 'effnetv2':
            print('ddp: Fetal_frame')
            dataset = Fetal_frame(root = config.DATA.DATA_PATH, ann_path = ann_path, transform = transform, img_scaling = config.DATA.IMG_SCALING)
        nb_classes = config.MODEL.NUM_CLASSES
    if config.EVAL_MODE:
        if config.MODEL.TASK_TYPE == 'reg':
            print('regression: Fetal_frame_eval_reg')
            if config.MODEL.TYPE == 'effnetv2_key_frame' or config.MODEL.TYPE == 'swin-video':
                print(config.MODEL.TYPE)
                print('ddp: Video_Loader')
                dataset = Eval_Video_Loader(root = config.DATA.DATA_PATH, ann_path = ann_path, 
                                       transform = transform, img_scaling = config.DATA.IMG_SCALING, num_frames = config.TRAIN.NUM_FRAMES, img_size = config.DATA.IMG_SIZE)
            else:
                dataset = Fetal_frame_eval_reg(root = config.DATA.DATA_PATH, ann_path = ann_path, transform = transform)
                nb_classes = config.MODEL.NUM_CLASSES
        elif config.MODEL.TASK_TYPE == 'cls':
            print('classification: Fetal_frame_eval_cls')
            dataset = Fetal_frame_eval_cls(root = config.DATA.DATA_PATH, ann_path = ann_path, transform = transform)
            nb_classes = config.MODEL.NUM_CLASSES



    return dataset, nb_classes


def build_transform(is_train, config, dataset_name):
    if dataset_name == 'fetal':
        if is_train:
            if config.DATA.AUGM:
                            t = transforms.Compose([
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomAffine(degrees=5, scale = (0.7, 1), fill = -0.7436),
                            ])
            else:
                t = None
                #transforms.Normalize(mean=0.1354949, std=0.18222201)
        else:
            t = None

    return t