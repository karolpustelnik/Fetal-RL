# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# For Comet to start tracking a training run,
# just add these two lines at the top of
# your training script:
import comet_ml

experiment = comet_ml.Experiment(
    api_key="3YfcpxE1bYPCpkkg4pQ2OjQ2r",
    project_name="Fetal swin cls"
)

# Metrics from this training run will now be
# available in the Comet UI
import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from torchmetrics.functional import precision_recall
from torchmetrics.functional import auc
from torchmetrics.functional import f1_score


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
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true', help='Fused window shift & window partition, similar for reversed part.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    model_without_ddp = model

    optimizer = build_optimizer(config, model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        print(f'{config.AUG.MIXUP} ##########################################################################')
        criterion_cls = SoftTargetCrossEntropy()
        criterion_reg = torch.nn.MSELoss()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        print(f'{config.MODEL.LABEL_SMOOTHING} ##########################################################################')
        criterion_cls = LabelSmoothingCrossEntropy()
        criterion_reg = torch.nn.MSELoss()
    else:
        print('dziala jak trzeba ##########################################################################')
        criterion_cls = torch.nn.CrossEntropyLoss()
        criterion_reg = torch.nn.MSELoss()

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
        acc1, f1_score, recall, precision,  loss_cls, loss_reg = validate(config, data_loader_val, model)
        if config.MODEL.TASK_TYPE == 'cls':
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            logger.info(f"f1 score of the network on the {len(dataset_val)} test images: {f1_score*100}%")
            logger.info(f"recall of the network on the {len(dataset_val)} test images: {recall*100}%")
            logger.info(f"precision of the network on the {len(dataset_val)} test images: {precision*100}%")
            logger.info(f"Loss of the network on the {len(dataset_val)} test images: {loss_cls:.4f}")
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        if config.MODEL.TASK_TYPE == 'reg':
            logger.info(f"Loss of the network on the {len(dataset_val)} test images: {loss_reg:.4f}")
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        acc1, f1_score, recall, precision,  loss_cls, loss_reg = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        logger.info(f"f1 score of the network on the {len(dataset_val)} test images: {f1_score*100}%")
        logger.info(f"recall of the network on the {len(dataset_val)} test images: {recall*100}%")
        logger.info(f"precision of the network on the {len(dataset_val)} test images: {precision*100}%")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion_cls, criterion_reg, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger)

        acc1, f1_score, recall, precision,  loss_cls, loss_reg = validate(config, data_loader_val, model)
        if config.MODEL.TASK_TYPE == 'cls':
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            logger.info(f"f1 score of the network on the {len(dataset_val)} test images: {f1_score*100}%")
            logger.info(f"recall of the network on the {len(dataset_val)} test images: {recall*100}%")
            logger.info(f"precision of the network on the {len(dataset_val)} test images: {precision*100}%")
            logger.info(f"Loss of the network on the {len(dataset_val)} test images: {loss_cls:.4f}")
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        if config.MODEL.TASK_TYPE == 'reg':
            logger.info(f"Loss of the network on the {len(dataset_val)} test images: {loss_reg:.4f}")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion_cls, criterion_reg, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter_cls = AverageMeter() ## loss for classifier
    loss_meter_reg = AverageMeter() ## loss for regression
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets, scores) in enumerate(data_loader): ## changed
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        scores = scores.type(torch.float16).cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets) ## changed

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
        if config.MODEL.TASK_TYPE == 'cls':
            loss_cls = criterion_cls(outputs, targets) ## changed
            loss = loss_cls
        if config.MODEL.TASK_TYPE == 'reg':
            loss_reg = criterion_reg(outputs, scores)
            loss = loss_reg
        if config.MODEL.TASK_TYPE == 'cls_reg':
            loss_cls = criterion_cls(outputs[0], targets)
            loss_reg = criterion_reg(outputs[1], scores)
            loss = loss_cls + loss_reg
        # this attribute is added by timm on one optimizer (adahessian)
        
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()
        if config.MODEL.TASK_TYPE == 'cls':
            loss_meter_cls.update(loss_cls.item(), targets.size(0))
        if config.MODEL.TASK_TYPE == 'reg':
            loss_meter_reg.update(loss_reg.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'cls loss {loss_meter_cls.val:.4f} ({loss_meter_cls.avg:.4f})\t'
                f'reg loss {loss_meter_reg.val:.4f} ({loss_meter_reg.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
            #commet ml logging
            experiment.log_metric("lr", lr)
            experiment.log_metric("wd", wd)
            experiment.log_metric("grad_norm", norm_meter.avg)
            experiment.log_metric("cls loss", loss_meter_cls.avg)
            experiment.log_metric("reg loss", loss_meter_reg.avg)
            
    epoch_time = time.time() - start
    
    # comet ml logging
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model):
    criterion_cls = torch.nn.CrossEntropyLoss() ## changed
    criterion_reg = torch.nn.MSELoss() ## changed
    model.eval()

    batch_time = AverageMeter()
    loss_meter_cls = AverageMeter()
    loss_meter_reg = AverageMeter()
    acc1_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    f1_score_meter = AverageMeter()

    end = time.time()
    for idx, (images, target, scores) in enumerate(data_loader): ## changed
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        scores = scores.type(torch.float16).cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        if config.MODEL.TASK_TYPE == 'cls':
            loss_cls = criterion_cls(output, target)
        if config.MODEL.TASK_TYPE == 'reg':
            loss_reg = criterion_reg(output, scores)
            
        if config.MODEL.TASK_TYPE == 'cls':
            acc1, _ = accuracy(output, target, topk=(1, 5))
            precision, recall = precision_recall(output, target, average = 'macro', num_classes = 7)
            f1 = f1_score(output, target, average = 'macro', num_classes = 7)
            acc1 = reduce_tensor(acc1)
            f1 = reduce_tensor(f1)
            precision = reduce_tensor(precision)
            recall = reduce_tensor(recall)
            loss_cls = reduce_tensor(loss_cls)
            loss_meter_cls.update(loss_cls.item(), target.size(0))
            acc1_meter.update(acc1.item(), target.size(0))
            precision_meter.update(precision.item(), target.size(0))
            recall_meter.update(recall.item(), target.size(0))
            f1_score_meter.update(f1.item(), target.size(0))
            
        if config.MODEL.TASK_TYPE == 'reg':
            loss_reg = reduce_tensor(loss_reg)
            loss_meter_reg.update(loss_reg.item(), target.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'cls Loss {loss_meter_cls.val:.4f} ({loss_meter_cls.avg:.4f})\t'
                f'reg Loss {loss_meter_reg.val:.4f} ({loss_meter_reg.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'f@1_score {f1_score_meter.val:.3f} ({f1_score_meter.avg:.3f})\t'
                f'recall {recall_meter.val:.3f} ({recall_meter.avg:.3f})\t'
                f'precision {precision_meter.val:.3f} ({precision_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f}')
    logger.info(f' * f@1_score {f1_score_meter.avg:.3f}')
    logger.info(f' * recall {recall_meter.avg:.3f}')
    logger.info(f' * precision {precision_meter.avg:.3f}')
    
    ## comet ml logging
    experiment.log_metric('test_accuracy', acc1_meter.avg)
    experiment.log_metric('test_f1_score', f1_score_meter.avg)
    experiment.log_metric('test_recall', recall_meter.avg)
    experiment.log_metric('test_precision', precision_meter.avg)
    experiment.log_metric('test_loss_cls', loss_meter_cls.avg)

    return acc1_meter.avg, f1_score_meter.avg, recall_meter.avg, precision_meter.avg, loss_meter_cls.avg, loss_meter_reg.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
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
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
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
    
    # comet ml logging
    experiment.log_parameters(vars(args))

    main(config)
