import wandb
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
from utils_buffer import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
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
    wandb.init(project="Fetal-Multimodal")#WANDB
        
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    detector, regresor = build_model(config)
    logger.info(str(detector))
    logger.info(str(regresor))
    detector.to('cuda:0')
    regresor.to('cuda:1')

    
    n_parameters_detector = sum(p.numel() for p in detector.parameters() if p.requires_grad)
    n_parameters_regresor = sum(p.numel() for p in regresor.parameters() if p.requires_grad)
    print(f"Number of trainable parameters of detector: {n_parameters_detector}")
    print(f"Number of trainable parameters of regresor: {n_parameters_regresor}")
    
    
    

    optimizer_detector = build_optimizer(config, detector)
    optimizer_regresor = build_optimizer(config, regresor)
    print('Optimizer built!')
    print(f'Local rank from environment: {local_rank}')
    print(f'Local rank from config: {config.LOCAL_RANK}')
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer_detector, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
        reg_lr_scheduler = build_scheduler(config, optimizer_regresor, 7_500 // config.TRAIN.ACCUMULATION_STEPS) # approx. length of dataset
    else:
        lr_scheduler = build_scheduler(config, optimizer_detector, len(data_loader_train))
        reg_lr_scheduler = build_scheduler(config, optimizer_regresor, 7_500) # approx. length of dataset

    criterion_cls = torch.nn.CrossEntropyLoss() ## changed
    criterion_reg = torch.nn.L1Loss() ## changed

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file_detector = auto_resume_helper(config.OUTPUT, 'detector')
        resume_file_regresor = auto_resume_helper(config.OUTPUT, 'regresor')
        if resume_file_detector and resume_file_regresor :
            config.defrost()
            config.MODEL.RESUME_DETECTOR = resume_file_detector
            config.MODEL.RESUME_REGRESOR = resume_file_regresor
            config.freeze()
            logger.info(f'auto resuming from {resume_file_detector} and {resume_file_regresor}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME_DETECTOR or config.MODEL.RESUME_REGRESOR:
        print('Loading checkpoint...')
        max_accuracy = load_checkpoint(config, detector, 'detector', optimizer_detector, lr_scheduler, loss_scaler, logger)
        max_accuracy2 = load_checkpoint(config, regresor, 'regresor', optimizer_regresor, reg_lr_scheduler, loss_scaler, logger)
        # print('Validating model after loading checkpoint...')
        # acc1, f1_score, recall, precision, loss_cls, loss_reg, mae_meter, mape_meter, rmse_meter, n_test_reg = validate(config, data_loader_val, dataset_val, detector, regresor)
        # logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        # logger.info(f"f1 score of the network on the {len(dataset_val)} test images: {f1_score*100}%")
        # logger.info(f"recall of the network on the {len(dataset_val)} test images: {recall*100}%")
        # logger.info(f"precision of the network on the {len(dataset_val)} test images: {precision*100}%")
        # logger.info(f"Loss of the network on the {len(dataset_val)} test images: {loss_cls:.4f}")
        # max_accuracy = max(max_accuracy, acc1)
        # logger.info(f"Loss of the network on the {n_test_reg} test images: {loss_reg:.4f}")
        # logger.info(f"MAE of the network on the {n_test_reg} test images: {mae_meter:.4f}")
        # logger.info(f"MAPE of the network on the {n_test_reg} test images: {mape_meter:.4f}")
        # logger.info(f"RMSE of the network on the {n_test_reg} test images: {rmse_meter:.4f}")
        if config.EVAL_MODE:
            return


    logger.info("Start training")
    start_time = time.time()
    if config.EVAL_MODE == False:
        
        for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
            data_loader_train.sampler.set_epoch(epoch) if config.PARALLEL_TYPE == 'ddp' else None

            train_one_epoch(config, detector, regresor, criterion_cls, criterion_reg, data_loader_train, 
                    optimizer_detector, optimizer_regresor,
                   epoch, lr_scheduler, reg_lr_scheduler, dataset_train)
            if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
               save_checkpoint(config, epoch, detector, 'detector', max_accuracy, optimizer_detector, lr_scheduler, loss_scaler,
                               logger)
               save_checkpoint(config, epoch, regresor, 'regresor', max_accuracy, optimizer_regresor, reg_lr_scheduler, loss_scaler,
                               logger)
            print(f'Validating model after {epoch}...')
            validate(config, data_loader_val, dataset_val, detector, regresor)

            
    elif config.EVAL_MODE == True:
        acc1, f1_score, recall, precision, loss_cls, loss_reg, mae_meter, mape_meter, rmse_meter, n_test_reg = validate(config, data_loader_val, dataset_val, detector, regresor)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        logger.info(f"f1 score of the network on the {len(dataset_val)} test images: {f1_score*100}%")
        logger.info(f"recall of the network on the {len(dataset_val)} test images: {recall*100}%")
        logger.info(f"precision of the network on the {len(dataset_val)} test images: {precision*100}%")
        logger.info(f"Cls Loss of the network on the {len(dataset_val)} test images: {loss_cls:.4f}")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f"Loss of the network on the {n_test_reg} test images: {loss_reg:.4f}")
        logger.info(f"MAE of the network on the {n_test_reg} test images: {mae_meter:.4f}")
        logger.info(f"MAPE of the network on the {n_test_reg} test images: {mape_meter:.4f}")
        logger.info(f"RMSE of the network on the {n_test_reg} test images: {rmse_meter:.4f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    
    
def train_one_epoch(config, detector, regresor, criterion_cls, criterion_reg, data_loader, 
                    optimizer_detector, optimizer_regresor,
                    epoch, lr_scheduler, reg_lr_scheduler, dataset_train):
    detector.train()
    regresor.train()

    num_steps = len(data_loader)
    loss_meter_cls = AverageMeter() 
    loss_meter_reg = AverageMeter()
    buffer = torch.tensor([], dtype=torch.int64)
    reg_epoch = 0
    buffer_size = config.DATA.BATCH_SIZE
    for idx, (images, Class, measure, ps, frames_n, measure_normalized, indexes) in enumerate(data_loader): ## changed
        images = images.to('cuda:0')
        if epoch < config.TRAIN.WARMUP_EPOCHS:
            buffer = torch.cat((buffer, indexes[Class == 2]))
            buffer = torch.cat((buffer, indexes[Class == 4]))
            buffer = torch.cat((buffer, indexes[Class == 6]))
        if (len(buffer) >= buffer_size) and (epoch < config.TRAIN.WARMUP_EPOCHS):
            reg_epoch += 1
            buffer = buffer.tolist()
            images_reg, Class_reg, measure_reg, ps_reg, frames_n_reg, measure_normalized_reg, indexes_reg = dataset_train.load_batch(buffer[:buffer_size])
            del buffer[:buffer_size] # delete first 32 elements
            buffer = torch.tensor(buffer, dtype=torch.int64)
            optimizer_regresor.zero_grad()
            reg_output = regresor(images_reg.to('cuda:1'))
            measure_normalized_reg = measure_normalized_reg.unsqueeze(0).to('cuda:1')
            measure_normalized_reg = measure_normalized_reg.reshape(-1, 1)
            
            reg_loss = criterion_reg(reg_output, measure_normalized_reg)
            loss_meter_reg.update(reg_loss.item())
            reg_loss.backward()
            optimizer_regresor.step()
            reg_lr_scheduler.step_update((epoch * 7_500 + reg_epoch))
            
        optimizer_detector.zero_grad()
        det_output = detector(images)
        predicted_classes = torch.nn.functional.softmax(det_output, dim=1).argmax(dim = 1).cpu()
        if epoch >= config.TRAIN.WARMUP_EPOCHS:
            buffer = torch.cat((buffer, indexes[predicted_classes == 2]))
            buffer = torch.cat((buffer, indexes[predicted_classes == 4]))
            buffer = torch.cat((buffer, indexes[predicted_classes == 6]))
        Class = Class.to('cuda:0')
        loss = criterion_cls(det_output, Class)
        loss_meter_cls.update(loss.item())
        loss.backward()
        optimizer_detector.step()
        lr_scheduler.step_update((epoch * num_steps + idx))
        if (len(buffer) >= buffer_size) and (epoch >= config.TRAIN.WARMUP_EPOCHS):
            reg_epoch += 1
            buffer = buffer.tolist()
            images_reg, Class_reg, measure_reg, ps_reg, frames_n_reg, measure_normalized_reg, indexes_reg = dataset_train.load_batch(buffer[:buffer_size])
            del buffer[:buffer_size] # delete first 32 elements
            buffer = torch.tensor(buffer, dtype=torch.int64)
            optimizer_regresor.zero_grad()
            reg_output = regresor(images_reg.to('cuda:1'))
            measure_normalized_reg = measure_normalized_reg.unsqueeze(0).to('cuda:1')
            measure_normalized_reg = measure_normalized_reg.reshape(-1, 1)
            
            reg_loss = criterion_reg(reg_output, measure_normalized_reg)
            loss_meter_reg.update(reg_loss.item())
            reg_loss.backward()
            optimizer_regresor.step()
            reg_lr_scheduler.step_update((epoch * 7_500 + reg_epoch))
            

        torch.cuda.synchronize()
            

        if idx % config.PRINT_FREQ == 0:
            lr_cls = optimizer_detector.param_groups[0]['lr']
            lr_reg = optimizer_regresor.param_groups[0]['lr']
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'cls loss {loss_meter_cls.val:.4f} ({loss_meter_cls.avg:.4f})\t'
                f'reg loss {loss_meter_reg.val:.4f} ({loss_meter_reg.avg:.4f})\t')
            wandb.log({"Train Loss": loss_meter_cls.avg, "Train Reg Loss": loss_meter_reg.avg, "lr_cls": lr_cls, "lr_reg": lr_reg})
            # add logging
    print(f"Epoch {epoch} finished. Loss: {loss_meter_cls.avg:.4f}, Reg Loss: {loss_meter_reg.avg:.4f}. Number of reg epochs: {reg_epoch}; seen img: {reg_epoch * buffer_size}")

            
            
            
@torch.no_grad()
def validate(config, data_loader, dataset_val, detector, regresor):

    criterion_cls = torch.nn.CrossEntropyLoss() ## changed
    criterion_reg = torch.nn.L1Loss() ## changed
    mae = MeanAbsoluteError().to('cuda:1')
    mape = MeanAbsolutePercentageError().to('cuda:1')
    rmse = MeanSquaredError(squared = False).to('cuda:1')
    
    detector.eval()
    regresor.eval()
    
    loss_meter_cls = AverageMeter()
    loss_meter_reg = AverageMeter()
    acc1_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    f1_score_meter = AverageMeter()
    results = torch.tensor([])
    frames = []
    videos = []
    measures = []
    Classes = []
    mae_meter = AverageMeter()
    mape_meter = AverageMeter()
    rmse_meter = AverageMeter()
    reg_epoch = 0
    buffer = torch.tensor([], dtype=torch.int64)
    buffer_size = config.DATA.BATCH_SIZE
    #buffer_size = config.DATA.BATCH_SIZE
    #buffer = torch.tensor([], dtype=torch.int64)
    for idx, (images, Class, measure, ps, frames_n, measure_normalized, indexes) in enumerate(data_loader):

        det_output = detector(images.to('cuda:0'))
        predicted_classes = torch.nn.functional.softmax(det_output, dim=1).argmax(dim = 1).cpu()
        buffer = torch.cat((buffer, indexes[predicted_classes == 2]))
        buffer = torch.cat((buffer, indexes[predicted_classes == 4]))
        buffer = torch.cat((buffer, indexes[predicted_classes == 6]))
        Class = Class.to('cuda:0')
        loss_cls = criterion_cls(det_output, Class)
        loss_meter_cls.update(loss_cls.item())
        acc1 = accuracy(det_output, Class, num_classes=config.MODEL.NUM_CLASSES, average = 'micro', 
                        mdmc_average = 'global' if config.PARALLEL_TYPE == 'model_parallel' else None)
        precision, recall = precision_recall(det_output, Class, num_classes=config.MODEL.NUM_CLASSES, average = 'micro',
                                                mdmc_average = 'global' if config.PARALLEL_TYPE == 'model_parallel' else None)
        f1 = f1_score(det_output, Class, num_classes=config.MODEL.NUM_CLASSES, average = 'micro', 
                        mdmc_average = 'global' if config.PARALLEL_TYPE == 'model_parallel' else None)
        
        loss_meter_cls.update(loss_cls.item())
        acc1_meter.update(acc1.item())
        precision_meter.update(precision.item())
        recall_meter.update(recall.item())
        f1_score_meter.update(f1.item())
        
        
        if (len(buffer) >= buffer_size):
            reg_epoch += 1
            buffer = buffer.tolist()
            images_reg, Class_reg, measure_reg, ps_reg, frames_n_reg, measure_normalized_reg, indexes_reg = dataset_val.load_batch(buffer[:buffer_size])
            del buffer[:buffer_size] # delete first 32 elements
            buffer = torch.tensor(buffer, dtype=torch.int64)
            reg_output = regresor(images_reg.to('cuda:1'))
            measure_normalized_reg = measure_normalized_reg.unsqueeze(0).to('cuda:1')
            measure_normalized_reg = measure_normalized_reg.reshape(-1, 1)
            reg_loss = criterion_reg(reg_output, measure_normalized_reg)
            loss_meter_reg.update(reg_loss.item())
            ps = ps.unsqueeze(1)
            ps = ps.cpu().numpy()
            scaler = joblib.load('/data/kpusteln/Fetal-RL/data_preparation/scripts/scaler_filename')
            predicted_measure = scaler.inverse_transform(reg_output.cpu().numpy()) * ps
            predicted_measure = torch.from_numpy(predicted_measure)
            predicted_measure = predicted_measure.to('cuda:1')
            #predicted_measure = outputs * max_measure * ps
            predicted_measure = predicted_measure.squeeze(1)
            measure = measure.to('cuda:1')
            mae_value = mae(predicted_measure, measure)
            mape_value = mape(predicted_measure, measure)
            rmse_value = rmse(predicted_measure, measure)
            #print(f"Predicted measure: {predicted_measure[0]}, real measure: {measure[0]}, mae: {mae_value}, mape: {mape_value}, rmse: {rmse_value}")
            mae_meter.update(mae_value)
            mape_meter.update(mape_value)
            rmse_meter.update(rmse_value)
    
        
        # acc1 = reduce_tensor(acc1)
        # print(f'acc1 after reducing tensor: {acc1}')
        # f1 = reduce_tensor(f1)
        # precision = reduce_tensor(precision)
        # recall = reduce_tensor(recall)
        # loss_cls = reduce_tensor(loss_cls)


        if idx % config.PRINT_FREQ == 0:
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'reg Loss {loss_meter_reg.val:.4f} ({loss_meter_reg.avg:.4f})\t' 
                f'cls Loss {loss_meter_cls.val:.4f} ({loss_meter_cls.avg:.4f})\t' 
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t' 
                f'f@1_score {f1_score_meter.val:.3f} ({f1_score_meter.avg:.3f})\t'
                f'recall {recall_meter.val:.3f} ({recall_meter.avg:.3f})\t' 
                f'precision {precision_meter.val:.3f} ({precision_meter.avg:.3f})\t')
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'reg Loss {loss_meter_reg.val:.4f} ({loss_meter_reg.avg:.4f})\t' 
                f'mae {mae_meter.val:.3f} ({mae_meter.avg:.3f})\t' 
                f'mape {mape_meter.val:.3f} ({mape_meter.avg:.3f})\t' 
                f'rmse {rmse_meter.val:.3f} ({rmse_meter.avg:.3f})\t')
    print('Finished validation! Results:')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f}')
    logger.info(f' * f@1_score {f1_score_meter.avg:.3f}')
    logger.info(f' * recall {recall_meter.avg:.3f}')
    logger.info(f' * precision {precision_meter.avg:.3f}')
    logger.info(f' * mae {mae_meter.avg:.3f}')
    logger.info(f' * mape {mape_meter.avg:.3f}')
    logger.info(f' * rmse {rmse_meter.avg:.3f}')
    logger.info(f' * Reg loss {loss_meter_reg.avg:.3f}')
    wandb.log({'test_acc': acc1_meter.avg, 'test_f1': f1_score_meter.avg, 'test_recall': recall_meter.avg, 'test_precision': precision_meter.avg, 'test_mae': mae_meter.avg, 'test_mape': mape_meter.avg, 'test_rmse': rmse_meter.avg, 'test_reg_loss': loss_meter_reg.avg})
    n_test_reg = reg_epoch * buffer_size

    return acc1_meter.avg, f1_score_meter.avg, recall_meter.avg, precision_meter.avg, loss_meter_cls.avg, loss_meter_reg.avg, mae_meter.avg, mape_meter.avg, rmse_meter.avg, n_test_reg



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
