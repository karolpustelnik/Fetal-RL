import wandb
import warnings
warnings.filterwarnings("ignore")
import hashlib
import os
import time
import json
import random
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import AverageMeter
from torchmetrics.functional import precision_recall, accuracy
from torchmetrics.functional import auc
from torchmetrics.functional import f1_score
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanAbsolutePercentageError
from torchmetrics import MeanSquaredError
from torchmetrics import R2Score
from torchmetrics.functional import r2_score
import joblib
from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from new_utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor


def setup(rank, world_size):
    """
    Args:
        rank: current process rank
        world_size: number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    if world_size > 1:
        port = '29500'
    else:
        port = str(random.randint(100, 60000))
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group(backend = "gloo", rank=rank, world_size=world_size)
    
    

def parse_option():
    parser = argparse.ArgumentParser('Training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, default='configs/yamls/effnetv2_reg_img_scaling.yaml', metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch_size', type=int, help="batch size for single GPU")
    parser.add_argument('--data_path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel')
    parser.add_argument("--train_path", type=str, help='path to training data')
    parser.add_argument("--val_path", type=str, help='path to validation data')
    parser.add_argument("--path_prefix", type=str, help='path prefix eq. /home/karol')
    # for acceleration
    parser.add_argument("--augm", help='use augmentation')
    parser.add_argument("--dropout", type = float, help='dropout')
    parser.add_argument("--base_lr", type = float, help='base learning rate')
    parser.add_argument("--optimizer", type = str, help='optimizer')
    parser.add_argument("--scaling", help='whether to use imgscaling')
    parser.add_argument("--sigmoid", help='whether to use sigmoid')
    parser.add_argument("--attention", help='whether to use spatial attention')
    parser.add_argument("--loss", type = str, help='loss function')
    parser.add_argument('--key_frame_attention', help='whether to use key frame attention')
    parser.add_argument('--num_frames', type = int, help='num frames to use during training')
    parser.add_argument('--use_alpha', help = 'whether to use alpha in skip connection')
    parser.add_argument('--use_skip_connection', help = 'whether to use_skip_connection in attention modules')
    parser.add_argument('--use_gelu', help = 'whether to use gelu in attention modules')
    parser.add_argument('--use_batch_norm', help = 'whether to use batch norm in attention modules')
    parser.add_argument('--use_head', help = 'whether to use regression head in model')
    parser.add_argument('--backbone', type = str,  help = 'what bacbkone to use')
    parser.add_argument('--img_size', type = int, help = 'size of input img')
    parser.add_argument("--weight_decay", type = float, help='r2 regularization')
    
    #parser.add_argument('--num_workers', type = int, help='number of workers to use in dataloader')
    
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config



 
def main(rank, world_size, config):
    setup(rank, world_size)
    print("--------------rank", rank)
    print("---------------world size", world_size)
    if rank == 0:
        wandb.init() 
    seed = config.SEED + dist.get_rank()
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    #np.random.seed(seed)
    #random.seed(seed)
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = float(config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * np.sqrt(dist.get_world_size()))
    linear_scaled_warmup_lr = float(config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * np.sqrt(dist.get_world_size()))
    linear_scaled_min_lr = float(config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * np.sqrt(dist.get_world_size()))
    name = config.MODEL.NAME + f"bacbkone_{config.MODEL.BACKBONE}_body_part{config.MODEL.BODY_PART}_weight_decay_{config.TRAIN.WEIGHT_DECAY}_bs{config.DATA.BATCH_SIZE}_lr{config.TRAIN.BASE_LR}_drop{config.MODEL.DROP_RATE}_n_frames{config.TRAIN.NUM_FRAMES}_key_frame_att{config.MODEL.KEY_FRAME_ATTENTION}_alpha{config.MODEL.USE_ALPHA}_use_skip_connection{config.MODEL.USE_SKIP_CONNECTION}_use_gelu{config.MODEL.USE_GELU}_use_batch_norm{config.MODEL.USE_BATCH_NORM}_use_head{config.MODEL.USE_HEAD}"
    output = os.path.join(config.OUTPUT, name, config.TAG)
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.MODEL.NAME = name
    config.OUTPUT = output
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_parameters}")
    
    if config.PARALLEL_TYPE == 'ddp':
        model.to(rank)
    
    torch.cuda.set_device(rank)
    optimizer = build_optimizer(config, model)
    print('Optimizer built!')
    if config.PARALLEL_TYPE == 'ddp':
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False)
        print('Model wrapped in DDP!')
    loss_scaler = NativeScalerWithGradNormCount()
    print('data loader lenght', len(data_loader_train))
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train)/config.DATA.BATCH_SIZE)
    prefix = config.DATA.PATH_PREFIX
    #normed_weight = torch.load('/home/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/precision_weights.pt').cuda()
    if config.MODEL.BODY_PART == 'abdomen':
        weights = torch.load(f'{prefix}/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/biometry_scaled_ps/class_data_split/abdomen_class_weight.pt').cuda()
    elif config.MODEL.BODY_PART == 'head':
        weights = torch.load(f'{prefix}/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/biometry_scaled_ps/class_data_split/head_class_weight.pt').cuda()
    elif config.MODEL.BODY_PART == 'femur':
        weights = torch.load(f'{prefix}/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/biometry_scaled_ps/class_data_split/femur_class_weight.pt').cuda()
    elif config.MODEL.BODY_PART == 'all':
        weights = torch.load(f'{prefix}/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/biometry_scaled_ps/class_data/weights_all.pt').cuda()
    criterion_cls = torch.nn.CrossEntropyLoss(weight = weights)## changed
    criterion_reg = torch.nn.MSELoss() if config.MODEL.LOSS == 'L2' else torch.nn.L1Loss()

    max_accuracy = 0.0
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger)
        if config.MODEL.TASK_TYPE == 'cls':
            print('Validating model after loading checkpoint...')
            acc, f1, recall, precision, loss_cls = validate(config, data_loader_val, model, logger)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc:.1f}%")
            logger.info(f"f1 score of the network on the {len(dataset_val)} test images: {f1*100}%")
            logger.info(f"recall of the network on the {len(dataset_val)} test images: {recall*100}%")
            logger.info(f"precision of the network on the {len(dataset_val)} test images: {precision*100}%")
            logger.info(f"Loss of the network on the {len(dataset_val)} test images: {loss_cls:.4f}")
        elif config.MODEL.TASK_TYPE == 'reg':
            print('Validating model after loading checkpoint...')
            mae_meter, mape_meter, rmse_meter, loss_meter_reg = validate(config, data_loader_val, model, logger)
            logger.info(f"Loss of the network on the {len(dataset_val)} test images: {loss_meter_reg:.4f}")
            logger.info(f"MAE of the network on the {len(dataset_val)} test images: {mae_meter:.4f}")
            logger.info(f"MAPE of the network on the {len(dataset_val)} test images: {mape_meter:.4f}")
            logger.info(f"RMSE of the network on the {len(dataset_val)} test images: {rmse_meter:.4f}")
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model, logger)
        if config.MODEL.TASK_TYPE == 'cls':
            print('Validating model after loading pretrained weights...')
            acc1, f1_score, recall, precision, loss_cls = validate(config, data_loader_val, model, logger)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            logger.info(f"f1 score of the network on the {len(dataset_val)} test images: {f1_score*100}%")
            logger.info(f"recall of the network on the {len(dataset_val)} test images: {recall*100}%")
            logger.info(f"precision of the network on the {len(dataset_val)} test images: {precision*100}%")
        elif config.MODEL.TASK_TYPE == 'reg':
            print('Validating model after loading pretrained weights...')
            mae_meter, mape_meter, rmse_meter, loss_meter_reg = validate(config, data_loader_val, model, logger)
            logger.info(f"Loss of the network on the {len(dataset_val)} test images: {loss_meter_reg:.4f}")
            logger.info(f"MAE of the network on the {len(dataset_val)} test images: {mae_meter:.4f}")
            logger.info(f"MAPE of the network on the {len(dataset_val)} test images: {mape_meter:.4f}")
            logger.info(f"RMSE of the network on the {len(dataset_val)} test images: {rmse_meter:.4f}")

    logger.info("Start training")
    start_time = time.time()
    if config.EVAL_MODE == False:
        for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
            data_loader_train.sampler.set_epoch(epoch) if config.PARALLEL_TYPE == 'ddp' else None
            train_one_epoch(config, model, criterion_cls, criterion_reg, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                            loss_scaler, logger)
            if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                                logger)
            if config.MODEL.TASK_TYPE == 'cls':
                acc, f1, recall, precision, loss_cls = validate(config, data_loader_val, model, logger)
                if rank == 0:
                    wandb.log({'val_acc': acc, 'val_f1_score': f1, 'val_recall': recall, 'val_precision': precision}, step = epoch)
            elif config.MODEL.TASK_TYPE == 'reg':
                if epoch%15 == 0:
                    mae_meter, mape_meter, rmse_meter, loss_meter_reg = validate(config, data_loader_val, model, logger)
                    if rank == 0:
                        wandb.log({'val_mae': mae_meter, 'val_mape': mape_meter, 'val_rmse': rmse_meter, 'val_loss': loss_meter_reg}, step = epoch)
            
    elif config.EVAL_MODE == True:
        acc, f1, recall, precision, loss_cls = validate(config, data_loader_val, model, logger)
        if config.MODEL.TASK_TYPE == 'cls':
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            logger.info(f"f1 score of the network on the {len(dataset_val)} test images: {f1_score*100}%")
            logger.info(f"recall of the network on the {len(dataset_val)} test images: {recall*100}%")
            logger.info(f"precision of the network on the {len(dataset_val)} test images: {precision*100}%")
            logger.info(f"Loss of the network on the {len(dataset_val)} test images: {loss_cls:.4f}")
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        elif config.MODEL.TASK_TYPE == 'reg':
            logger.info(f"Loss of the network on the {len(dataset_val)} test images: {loss_reg:.4f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    
    
def train_one_epoch(config, model, criterion_cls, criterion_reg, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, logger):
    model.train()

    num_steps = len(data_loader)
    batch_time = AverageMeter() # stores average and current value
    loss_meter_cls = AverageMeter() 
    loss_meter_reg = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    for idx, (images, Class, measure, ps, frames_n, measure_scaled, index, days_normalized, frame_loc, measure_normalized, org_seq_lens) in enumerate(data_loader): ## changed
        optimizer.zero_grad()
        if config.PARALLEL_TYPE == 'ddp':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                
                if config.MODEL.TYPE == 'effnetv2_meta':
                    meta = torch.stack((days_normalized, frame_loc), dim = 1).cuda(non_blocking=True)
                    outputs = model((images, meta)) 
                elif config.MODEL.TYPE == 'effnetv2_key_frame':
                        #images = images.squeeze(0)
                        #('shape of images', images.shape)
                        outputs = model(images, org_seq_lens)
                elif config.MODEL.TYPE == 'swin-video':
                    org_batch_unsqueezed = [i.unsqueeze(0) for i in images]
                    org_batch_unsqueezed = torch.cat(org_batch_unsqueezed, dim = 0)
                    org_batch_unsqueezed = org_batch_unsqueezed.permute(0, 2, 1, 3, 4)
                    outputs = model(org_batch_unsqueezed)
                elif config.MODEL.TYPE == 'effnetv2':
                    outputs = model(images)
        elif config.PARALLEL_TYPE == 'model_parallel':
            #labels = labels.to('cuda:1')
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model((images, frames_n))

        
        if config.MODEL.TASK_TYPE == 'cls':
            Class = Class.cuda(non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):      
                loss = criterion_cls(outputs, Class) ## changed
            
        elif config.MODEL.TASK_TYPE == 'reg':
            measures_train = measure_normalized if config.DATA.IMG_SCALING else measure_scaled
            measures_train = measures_train.unsqueeze(0).cuda(non_blocking=True)
            measures_train = measures_train.reshape(-1, 1)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = criterion_reg(outputs, measures_train)
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step_update((epoch * num_steps + idx))

        torch.cuda.synchronize()
        if config.MODEL.TASK_TYPE == 'cls':
            loss_meter_cls.update(loss.item())
            
        elif config.MODEL.TASK_TYPE == 'reg':
            loss_meter_reg.update(loss.item())

            
        #if idx % config.PRINT_FREQ == 0:
        if idx == num_steps - 1:
            lr = optimizer.param_groups[0]['lr']
            if config.MODEL.TASK_TYPE == 'cls':
                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'cls loss {loss_meter_cls.val:.4f} ({loss_meter_cls.avg:.4f})\t'
                    f'lr {lr:.6f}\t')
                if dist.get_rank() == 0:
                    wandb.log({"Train Loss": loss_meter_cls.avg, "lr": lr})
            elif config.MODEL.TASK_TYPE == 'reg':
                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'reg loss {loss_meter_reg.val:.4f} ({loss_meter_reg.avg:.4f})\t'
                    f'lr {lr:.6f}\t')
                if dist.get_rank() == 0:
                    wandb.log({"Train Loss": loss_meter_reg.avg, "lr": lr})
            # add logging
            
    
            
            
            
@torch.no_grad()
def validate(config, data_loader, model, logger):
    #normed_weight = torch.load('/home/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/normedWeights.pt').cuda()
    prefix = config.DATA.PATH_PREFIX
    if config.MODEL.BODY_PART == 'abdomen':
        weights = torch.load(f'{prefix}/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/biometry_scaled_ps/class_data_split/abdomen_class_weight.pt').cuda()
    elif config.MODEL.BODY_PART == 'head':
        weights = torch.load(f'{prefix}/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/biometry_scaled_ps/class_data_split/head_class_weight.pt').cuda()
    elif config.MODEL.BODY_PART == 'femur':
        weights = torch.load(f'{prefix}/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/biometry_scaled_ps/class_data_split/femur_class_weight.pt').cuda()
    elif config.MODEL.BODY_PART == 'all':
        weights = torch.load(f'{prefix}/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/biometry_scaled_ps/class_data/weights_all.pt').cuda()
    criterion_cls = torch.nn.CrossEntropyLoss(weight = weights) ## changed
    criterion_reg = torch.nn.MSELoss() if config.MODEL.LOSS == 'L2' else torch.nn.L1Loss()
    mae = MeanAbsoluteError().cuda()
    mape = MeanAbsolutePercentageError().cuda()
    rmse = MeanSquaredError(squared = False).cuda()
    
    model.eval()
    
    loss_meter_cls = AverageMeter()
    loss_meter_reg = AverageMeter()
    # acc1_meter = AverageMeter()
    # precision_meter = AverageMeter()
    # recall_meter = AverageMeter()
    # f1_score_meter = AverageMeter()
    Classes = torch.tensor([]).cuda()
    predicts = torch.tensor([]).cuda()
    mae_meter = AverageMeter()
    mape_meter = AverageMeter()
    rmse_meter = AverageMeter()
    if config.MODEL.BODY_PART == 'abdomen':
        weights = torch.load(f'{prefix}/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/biometry_scaled_ps/class_data_split/abdomen_class_weight.pt').cuda()
    elif config.MODEL.BODY_PART == 'head':
        weights = torch.load(f'{prefix}/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/biometry_scaled_ps/class_data_split/head_class_weight.pt').cuda()
    elif config.MODEL.BODY_PART == 'femur':
        weights = torch.load(f'{prefix}/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/biometry_scaled_ps/class_data_split/femur_class_weight.pt').cuda()
    elif config.MODEL.BODY_PART == 'all':
        weights = torch.load(f'{prefix}/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/biometry_scaled_ps/class_data/weights_all.pt').cuda()
    for idx, (images, Class, measure, ps, frames_n, measure_scaled, index, days_normalized, frame_loc, measure_normalized, org_seq_lens) in enumerate(data_loader):
        #images = images.to(torch.float32)
        if config.PARALLEL_TYPE == 'ddp':
            #labels = labels.cuda(non_blocking=True)
            #print(labels.shape)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                
                if config.MODEL.TYPE == 'effnetv2_meta':
                    meta = torch.stack((days_normalized, frame_loc), dim = 1).cuda(non_blocking=True)
                    outputs = model((images, meta)) 
                elif config.MODEL.TYPE == 'effnetv2_key_frame':
                    #images = images.squeeze(0)
                    outputs = model(images, org_seq_lens)
                elif config.MODEL.TYPE == 'swin-video':
                    org_batch_unsqueezed = [i.unsqueeze(0) for i in images]
                    org_batch_unsqueezed = torch.cat(org_batch_unsqueezed, dim = 0)
                    org_batch_unsqueezed = org_batch_unsqueezed.permute(0, 2, 1, 3, 4)
                    outputs = model(org_batch_unsqueezed)
                elif config.MODEL.TYPE == 'effnetv2':
                    outputs = model(images)
            
        elif config.PARALLEL_TYPE == 'model_parallel':
            #labels = labels.to('cuda:1')
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model((images, frames_n))

        # measure accuracy and record loss
        if config.MODEL.TASK_TYPE == 'cls':
            Class = Class.cuda(non_blocking=True)
            Classes = torch.cat((Classes, Class), dim = 0)
            predict = outputs.softmax(dim = 1).max(dim = 1)[1].to(torch.int64).cuda(non_blocking=True)
            predicts = torch.cat((predicts, predict), dim = 0)
            
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss_cls = criterion_cls(outputs, Class)
            
            if config.PARALLEL_TYPE == 'ddp':
                loss_cls = reduce_tensor(loss_cls)
            loss_meter_cls.update(loss_cls.item())
            
        elif config.MODEL.TASK_TYPE == 'reg':
            measures_train = measure_normalized if config.DATA.IMG_SCALING else measure_scaled
            measures_train = measures_train.unsqueeze(0).cuda(non_blocking=True)
            measures_train = measures_train.reshape(-1, 1)
            #print('outputs shape', outputs.shape)
            #print('outputs', outputs)
            #with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss_reg = criterion_reg(outputs, measures_train)
            #ps = ps.cuda(non_blocking=True)
            ps = ps.unsqueeze(1)
            ps = ps.cpu().numpy()
            prefix = config.DATA.PATH_PREFIX
            if config.DATA.IMG_SCALING:
                scaler = joblib.load(f'{prefix}/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/biometry_scaled_ps/normalizer_measure_img_scaling')
                predicted_measure = scaler.inverse_transform(outputs.cpu().numpy())
            else:
                if config.MODEL.BODY_PART == 'head':
                    scaler = joblib.load(f'{prefix}/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/biometry_scaled_ps/head/head_scaler')
                elif config.MODEL.BODY_PART == 'abdomen':
                    scaler = joblib.load(f'{prefix}/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/biometry_scaled_ps/abdomen/abdomen_scaler')
                elif config.MODEL.BODY_PART == 'femur':
                    scaler = joblib.load(f'{prefix}/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/biometry_scaled_ps/femur/femur_scaler')
                elif config.MODEL.BODY_PART == 'all':
                    scaler = joblib.load(f'{prefix}/kpusteln/Fetal-RL/data_preparation/data_biometry/ete_model/biometry_scaled_ps/all/all_scaler')
                predicted_measure = scaler.inverse_transform(outputs.cpu().numpy()) * ps
            predicted_measure = torch.from_numpy(predicted_measure)
            predicted_measure = predicted_measure.cuda(non_blocking=True)
            #predicted_measure = outputs * max_measure * ps
            predicted_measure = predicted_measure.squeeze(1)
            measure = measure.cuda(non_blocking=True)
            mae_value = mae(predicted_measure, measure)
            mape_value = mape(predicted_measure, measure)
            rmse_value = rmse(predicted_measure, measure)
            
            if config.PARALLEL_TYPE == 'ddp':
                loss_reg = reduce_tensor(loss_reg)
                mae_value = reduce_tensor(mae_value)
                mape_value = reduce_tensor(mape_value)
                rmse_value = reduce_tensor(rmse_value)
            
            mae_meter.update(mae_value)
            mape_meter.update(mape_value)
            rmse_meter.update(rmse_value)
            loss_meter_reg.update(loss_reg.item())
            
            

        #if idx % config.PRINT_FREQ == 0:
        if idx == len(data_loader) - 1:
            if config.MODEL.TASK_TYPE == 'cls':
                
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'cls Loss {loss_meter_cls.val:.4f} ({loss_meter_cls.avg:.4f})\t' )
                    # f'Acc@1 {acc:.3f}\t' 
                    # f'f@1_score {f1:.3f}\t'
                    # f'recall {recall:.3f}\t' 
                    # f'precision {precision:.3f}\t')
            elif config.MODEL.TASK_TYPE == 'reg':
                print(f"Example measure prediction: {predicted_measure[0]}, real measure: {measure[0]}, mae: {mae_value}, mape: {mape_value}, rmse: {rmse_value}")
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'reg Loss {loss_meter_reg.val:.4f} ({loss_meter_reg.avg:.4f})\t' 
                    f'mae {mae_meter.val:.3f} ({mae_meter.avg:.3f})\t' 
                    f'mape {mape_meter.val:.3f} ({mape_meter.avg:.3f})\t' 
                    f'rmse {rmse_meter.val:.3f} ({rmse_meter.avg:.3f})\t')
    
    if config.MODEL.TASK_TYPE == 'cls':
        predicts = reduce_tensor(predicts)
        Classes = reduce_tensor(Classes)
        acc = accuracy(predicts, Classes.int(), average = 'macro', num_classes=config.MODEL.NUM_CLASSES)
        precision, recall = precision_recall(predicts, Classes.int(), average = 'macro', 
                                            multiclass = True ,num_classes=config.MODEL.NUM_CLASSES)
        precision_per_class, recall_per_class = precision_recall(predicts, Classes.int(), average = None, num_classes=config.MODEL.NUM_CLASSES, multiclass = True)
        f1 = f1_score(predicts, Classes.int(), average='macro', multiclass= True, num_classes=config.MODEL.NUM_CLASSES)
        print('Recall and precision per class:', recall_per_class, precision_per_class)
        print('Finished validation! Results:')
        logger.info(f' * Acc@1 {acc:.3f}')
        logger.info(f' * f@1_score {f1:.3f}')
        logger.info(f' * recall {recall:.3f}')
        logger.info(f' * precision {precision:.3f}')
        logger.info(f' * cls loss {loss_meter_cls.avg:.3f}')
        if dist.get_rank() == 0:
            wandb.log({"Val Loss": loss_meter_cls.avg,})
            wandb.log({"Val Acc": acc,})
            wandb.log({"Val F1": f1,})
            wandb.log({"Val Recall": recall,})
            wandb.log({"Val Precision": precision,})
    elif config.MODEL.TASK_TYPE == 'reg':
        print('Finished validation! Results:')
        logger.info(f' * mae {mae_meter.avg:.3f}')
        logger.info(f' * mape {mape_meter.avg:.3f}')
        logger.info(f' * rmse {rmse_meter.avg:.3f}')
        logger.info(f' * Reg loss {loss_meter_reg.avg:.3f}')
        if dist.get_rank() == 0:
            wandb.log({"Val Loss": loss_meter_reg.avg,})
            wandb.log({"Val MAE": mae_meter.avg,})
            wandb.log({"Val MAPE": mape_meter.avg,})
            wandb.log({"Val RMSE": rmse_meter.avg,})
    
    if config.MODEL.TASK_TYPE == 'cls':
        return acc, f1, recall, precision, loss_meter_cls.avg
    elif config.MODEL.TASK_TYPE == 'reg':
        return mae_meter.avg, mape_meter.avg, rmse_meter.avg, loss_meter_reg.avg




if __name__ == '__main__':
    args, config = parse_option()
    world_size = torch.cuda.device_count()
    mp.spawn(main, nprocs=world_size, args=(world_size, config,))
