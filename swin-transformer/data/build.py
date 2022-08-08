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

from .fetal_loader import Fetal
from .elegans_loader import Elegans

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
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=config.TEST.SHUFFLE
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    affix = config.MODEL.AFFIX
    transform = build_transform(is_train, config, config.DATA.DATASET)
    if config.DATA.DATASET == 'fetal':
        prefix = 'fetal_extracted'
        if is_train:
            ann_path =  "/data/kpusteln/fetal/fetal_extracted_map_train_scr" + affix

        else:
            ann_path =  "/data/kpusteln/fetal/fetal_extracted_map_val_scr" + affix
        dataset = Fetal(config.DATA.DATA_PATH, ann_path, transform)
        nb_classes = config.MODEL.NUM_CLASSES
    elif config.DATA.DATASET == 'elegans':
        if is_train:
            ann_path =  "/data/kpusteln/elegans_fake/data_train.csv" #+ affix

        else:
            ann_path =  "/data/kpusteln/elegans_fake/data_test.csv" #+ affix
        dataset = Elegans(config.DATA.DATA_PATH, ann_path, transform)
        nb_classes = config.MODEL.NUM_CLASSES

    return dataset, nb_classes


def build_transform(is_train, config, dataset_name):
    if dataset_name == 'fetal':
        t = transforms.Compose([transforms.Resize((450, 600)),
                        transforms.Pad((0, 0, 0, 150), fill = 0, padding_mode = 'constant'),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor()])
    elif dataset_name == 'elegans':
        t = transforms.Compose([transforms.Pad((0, 0, 0, 176), fill = 0, padding_mode = 'constant'),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor()])
    return t