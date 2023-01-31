import wandb
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
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor
    


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
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

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true', help='Fused window shift & window partition, similar for reversed part.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config




def main(config):
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()#WANDB
    print('Initializing wandb logger...')#WANDB
    wandb.init(project="Fetal-Multimodal", name = config.MODEL.NAME)#WANDB
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_parameters}")
    
    if config.PARALLEL_TYPE == 'ddp':
        model.cuda()
    
    model_without_ddp = model

    optimizer = build_optimizer(config, model)
    print('Optimizer built!')
    print(f'Local rank from environment: {local_rank}')
    print(f'Local rank from config: {config.LOCAL_RANK}')
    if config.PARALLEL_TYPE == 'ddp':
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
        print('Model wrapped in DDP!')
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
        
    criterion_cls = torch.nn.CrossEntropyLoss() ## changed
    criterion_reg = torch.nn.L1Loss() ## changed

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
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        if config.MODEL.TASK_TYPE == 'cls':
            print('Validating model after loading pretrained weights...')
            acc1, f1_score, recall, precision,  loss_cls, loss_reg = validate(config, data_loader_val, model)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            logger.info(f"f1 score of the network on the {len(dataset_val)} test images: {f1_score*100}%")
            logger.info(f"recall of the network on the {len(dataset_val)} test images: {recall*100}%")
            logger.info(f"precision of the network on the {len(dataset_val)} test images: {precision*100}%")
        elif config.MODEL.TASK_TYPE == 'reg':
            print('Validating model after loading pretrained weights...')
            mae_meter, mape_meter, rmse_meter, loss_meter_reg = validate(config, data_loader_val, model)
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
                            loss_scaler)
            if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                                logger)
            if config.MODEL.TASK_TYPE == 'cls':
                acc1_meter, f1_score_meter, recall_meter, precision_meter = validate(config, data_loader_val, model)
                wandb.log({'val_acc': acc1_meter, 'val_f1_score': f1_score_meter, 'val_recall': recall_meter, 'val_precision': precision_meter}, step = epoch)
            elif config.MODEL.TASK_TYPE == 'reg':
                mae_meter, mape_meter, rmse_meter, loss_meter_reg = validate(config, data_loader_val, model)
                wandb.log({'val_mae': mae_meter, 'val_mape': mape_meter, 'val_rmse': rmse_meter, 'val_loss': loss_meter_reg}, step = epoch)
            
    elif config.EVAL_MODE == True:
        print('Evaluating model...')
        validate(config, data_loader_val, model)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    
    
def train_one_epoch(config, model, criterion_cls, criterion_reg, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    model.train()

    num_steps = len(data_loader)
    batch_time = AverageMeter() # stores average and current value
    loss_meter_cls = AverageMeter() 
    loss_meter_reg = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    
    start = time.time()
    end = time.time()
    for idx, (images, Class, measure, ps, frames_n, measure_normalized, indexes) in enumerate(data_loader): ## changed
        optimizer.zero_grad()
        
        if config.PARALLEL_TYPE == 'ddp':
            outputs = model(images)
        elif config.PARALLEL_TYPE == 'model_parallel':
            #labels = labels.to('cuda:1')
            print(f'images shape before forward: {images.shape}')
            print(f'frames_n shape before forward: {frames_n.shape}')
            outputs = model((images, frames_n))

        
        if config.MODEL.TASK_TYPE == 'cls':
            Class = Class.cuda(non_blocking=True)

            loss = criterion_cls(outputs, Class) ## changed
            
        elif config.MODEL.TASK_TYPE == 'reg':
            measure_normalized = measure_normalized.unsqueeze(0).cuda(non_blocking=True)
            measure_normalized = measure_normalized.reshape(-1, 1)
            loss = criterion_reg(outputs, measure_normalized)
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step_update((epoch * num_steps + idx))

        torch.cuda.synchronize()
        if config.MODEL.TASK_TYPE == 'cls':
            loss_meter_cls.update(loss.item())
            
        elif config.MODEL.TASK_TYPE == 'reg':
            loss_meter_reg.update(loss.item())

            

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            if config.MODEL.TASK_TYPE == 'cls':
                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'cls loss {loss_meter_cls.val:.4f} ({loss_meter_cls.avg:.4f})\t'
                    f'lr {lr:.6f}\t')
                wandb.log({"Train Loss": loss_meter_cls.avg, "lr": lr})
            elif config.MODEL.TASK_TYPE == 'reg':
                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'reg loss {loss_meter_reg.val:.4f} ({loss_meter_reg.avg:.4f})\t'
                    f'lr {lr:.6f}\t')
                wandb.log({"Train Loss": loss_meter_reg.avg, "lr": lr})
            # add logging
            
            
            
@torch.no_grad()
def validate(config, data_loader, model):
    criterion_cls = torch.nn.CrossEntropyLoss() ## changed
    criterion_reg = torch.nn.L1Loss() ## changed
    mae = MeanAbsoluteError().cuda()
    mape = MeanAbsolutePercentageError().cuda()
    rmse = MeanSquaredError(squared = False).cuda()
    
    model.eval()
    
    loss_meter_cls = AverageMeter()
    loss_meter_reg = AverageMeter()
    acc1_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    f1_score_meter = AverageMeter()
    
    mae_meter = AverageMeter()
    mape_meter = AverageMeter()
    rmse_meter = AverageMeter()

    worst_losses = []
    results = torch.tensor([])
    frames = []
    videos = []
    measures = []
    Classes = []
    pss = []
    batch = 0
    for idx, (images, frame_name, video, ps, Class) in enumerate(data_loader):
        batch += 1
        print(f'Batch {batch} out of {len(data_loader)}')
        if config.PARALLEL_TYPE == 'ddp':
            #labels = labels.cuda(non_blocking=True)
            #print(labels.shape)
            outputs = model(images) 
        if config.MODEL.TASK_TYPE == 'cls':
            predicted_classes = torch.nn.functional.softmax(outputs, dim=1).argmax(dim = 1).cpu()
            results = torch.cat((results, predicted_classes), 0)
            frames.extend(list(frame_name))
            videos.extend(list(video))
            ps = ps.cpu().numpy()
            pss.extend(list(ps))
            Class = Class.cpu().numpy()
            Classes.extend(list(Class))
        elif config.MODEL.TASK_TYPE == 'reg':
            ps = ps.unsqueeze(1)
            ps = ps.cpu().numpy()
            print(f'shape of ps :{ps.shape}')
            print(f'shape of outputs :{outputs.shape}')
            scaler = joblib.load('/data/kpusteln/Fetal-RL/data_preparation/scripts/scaler_filename')
            print(outputs)
            print(ps)
            predicted_measure = scaler.inverse_transform(outputs.cpu().numpy()) * ps
            print('Shape of predicted measure: ', predicted_measure.shape)
            #predicted_measure = outputs * max_measure * ps
            predicted_measure = predicted_measure.squeeze(1)
            measures.extend(predicted_measure)
            frames.extend(list(frame_name))
            videos.extend(list(video))
            
    if config.MODEL.TASK_TYPE == 'cls':
        results = results.to(torch.int64)
        #Classes = torch.stack(Classes)
        #Classes = Classes.numpy()
        data_frame = pd.DataFrame({'index': frames, 'video': videos, 'predict': results, 'gt': Classes, 'ps': pss})
        print('Saving...')
        data_frame.to_csv('/data/kpusteln/Fetal-RL/test_data/results_cls_two_models_val.csv', index = False)
        print('Finished!')
    elif config.MODEL.TASK_TYPE == 'reg':
        data_frame = pd.DataFrame({'index': frames, 'video': videos, 'measures': measures})
        print('Saving...')
        data_frame.to_csv('/data/kpusteln/Fetal-RL/test_data/results_reg_two_models_val_gt.csv', index = False)
        print('Finished!')
        

if __name__ == '__main__':
    args, config = parse_option()
    
    local_rank = int(os.environ["LOCAL_RANK"])
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    if config.PARALLEL_TYPE == 'ddp':
        torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0

    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
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
    logger.info(json.dumps(vars(args)))


    main(config)