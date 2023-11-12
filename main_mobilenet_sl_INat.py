import argparse
import math
import os
import shutil
import time
import random
from logging import getLogger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import apex
from apex.parallel.LARC import LARC
from torchvision import datasets, transforms
import sys
import wandb
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
)
from src.multicropdataset import MultiCropDataset
# import src.mobilenet_modified_swav as mobilenet_models
import src.mobilenet_modified_gelu2 as mobilenet_models
from tqdm import tqdm
# from mixup import FastCollateMixup, Mixup


'''
python3 main_mobilenet_sl.py --data_path /home/nina/ore_balanced_dataset --max-class 1960
'''

logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of MobileNetV3L SL")

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")
parser.add_argument("--use_pil_blur", type=bool_flag, default=True,
                    help="""use PIL library to perform blur instead of opencv""")

#########################
## swav specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--epsilon", default=0.05, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")
parser.add_argument("--improve_numerical_stability", default=False, type=bool_flag,
                    help="improves numerical stability in Sinkhorn-Knopp algorithm")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--feat_dim", default=128, type=int,
                    help="feature dimension") #128
parser.add_argument("--nmb_prototypes", default=3000, type=int,
                    help="number of prototypes")
parser.add_argument("--queue_length", type=int, default=0,
                    help="length of the queue (0 for no queue)")
parser.add_argument("--epoch_queue_starts", type=int, default=15,
                    help="from this epoch, we start using a queue")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=400, type=int,
                    help="number of total epochs to run") #400
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

#########################
#### other parameters ###
#########################
parser.add_argument("--arch", default="mobilenet_v3_large", type=str, help="convnet architecture")
parser.add_argument("--hidden_mlp", default=960, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--workers", default=8, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=25,
                    help="Save the model periodically")
parser.add_argument("--use_fp16", type=bool_flag, default=True,
                    help="whether to train with mixed precision or not")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=1993, help="seed") # seed=31
# number of classes used (imagenet training)
parser.add_argument('--max_class', default=100, type=int, help='number of classes used for training')

#wandb run name
parser.add_argument('--run_name', default='SL_balanced_data', type=str, help='name of wandb run')

parser.add_argument('--swav_aug', default=False, type=bool_flag, help='use swav augmentation')

parser.add_argument('--subset_size', default=50, type=int, help='subset size of current dataset')
parser.add_argument('--wandb', default=False, type=bool_flag, help='use wandb')
parser.add_argument('--image_net_normalize', default=False, type=bool_flag, help='normalize images to imagenet mean and std')
parser.add_argument('--lr_scheduler', default=True, type=bool_flag, help='use lr scheduler')
parser.add_argument('--sweep', default=False, type=bool_flag, help='use wandb sweep')

def main():
    torch.cuda.empty_cache()
    global args
    args = parser.parse_args()

    #assign class to subsize num_class
    args.max_class = args.subset_size

    #initiate wandb
    if args.wandb:
        wandb.init(project='mobilenet_sl', entity='ore-dreamteam', group=args.run_name, config=args)


    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")

    # build data
    if args.swav_aug:
        train_set =  datasets.ImageFolder(args.data_path)
        train_dataset = MultiCropDataset(
            args.data_path,
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
            pil_blur=args.use_pil_blur,
            return_label=True,
        )
        train_dataset.samples = train_set.samples
        print(len(train_dataset))
    else:
        input_shape = [3, 224, 224]

        if args.image_net_normalize:
            mean = [0.485, 0.456, 0.406]
            std = [0.228, 0.224, 0.225]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(input_shape[1:]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std),
        ])  
        train_dataset = datasets.ImageFolder(
            root=args.data_path,
            transform=train_transform
        )
        # print("labels", train_dataset.class_to_idx)
        
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    #wandb log first 10 images, each image is a list of images
    
    if args.wandb and args.swav_aug:
        if args.rank == 0:
            for image_list, labels in train_loader:
                image = image_list[0]
                images_to_log = [wandb.Image(image[i], caption=f'Label: {labels[i]}') for i in range(len(image))]
                wandb.log({'train_images': images_to_log})
                break
            #log local crop 

    # build model
    model = mobilenet_models.__dict__[args.arch](
        num_classes=args.max_class,
        pretrained=False)

    #set criterion:
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # synchronize batch norm layers
    if args.sync_bn == "pytorch":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif args.sync_bn == "apex":
        # with apex syncbn we sync bn per group because it speeds up computation
        # compared to global syncbn
        #apex depreacted
        # process_group = apex.parallel.create_syncbn_process_group(args.syncbn_process_group_size)
        # model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
        process_group = nn.create_syncbn_process_group(args.syncbn_process_group_size)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group=process_group)
    # copy model to GPU

    model = model.cuda()
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer >> SGD
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.875,
        weight_decay=args.wd,
    )

    #try adam
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=args.base_lr,
    #     weight_decay=args.wd,
    # )

    # LARC
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
            math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")

    # init mixed precision
    if args.use_fp16:
    #     #apex.amp deprecated
    #     model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        scaler = GradScaler()
        logger.info("Initializing mixed precision done.")

    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on],)
    

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    #restart_from_checkpoint(
    #    os.path.join(args.dump_path, "checkpoint.pth.tar"),
    #    run_variables=to_restore,
    #    state_dict=model,
    #    optimizer=optimizer,
    #    amp=apex.amp,
    #)
    start_epoch = to_restore["epoch"]

    # cudnn.benchmark = True
    
    # mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=0.5, num_classes=args.max_class)

    loss_epc=[]

    for epoch in range(start_epoch, args.epochs):
        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # train the network
        if args.lr_scheduler:
            lr = lr_schedule
        else:
            lr = args.base_lr
        
        scores, loss_avg, top1_avg = train(logger, criterion, train_loader, model, optimizer, epoch, lr, scaler=scaler)
        training_stats.update(scores)

        if args.rank == 0 and args.wandb:
            wandb.log({'epoch': epoch, 
                       'avg_loss': loss_avg,
                       'top1': top1_avg,
                       'lr': optimizer.param_groups[0]['lr']})

        loss_epc = np.append(loss_epc, loss_avg)

        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            # if args.use_fp16:
            #     save_dict["amp"] = apex.amp.state_dict()
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                np.save(os.path.join(args.dump_path, 'MobNet_train_loss.npy'), loss_epc)
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    os.path.join(args.dump_checkpoints, "MobNet_Ckpt_" + str(epoch) + ".pth"),
                )

def soft_cross_entropy(pred, soft_targets):
    return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))

def train(logger, criterion, train_loader, model, optimizer, epoch, lr_schedule, scaler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time() 

    # train_first_batch = train_loader.__iter__().__next__()
    # idx = 0
    # images, labels = train_first_batch

    # # image_list, labels = train_first_batch
    #     # Sepearte two batches for 2 high resolution crops, and the rest 6 for low resolution crops
    # # high_resolution_batch = image_list[:2]
    # # high_resolution_batch = torch.cat(high_resolution_batch, dim=0)
    # # high_resolution_batch.cuda(non_blocking=True)
    # # high_resolution_labels = labels.repeat(2)

    # # # #low resultion batch is the rest in image_list all extended, and should be tensor
    # # low_resolution_batch = image_list[2:]
    # # low_resolution_batch = torch.cat(low_resolution_batch, dim=0)
    # # low_resolution_batch.cuda(non_blocking=True)
    # # low_resolution_labels = labels.repeat(len(image_list)-2)

    # # labels = torch.cat((high_resolution_labels, low_resolution_labels), dim=0)


    # bsz = labels.shape[0]

    # iteration = epoch + idx
    # if args.lr_scheduler:
    #     for param_group in optimizer.param_groups:
    #         param_group["lr"] = lr_schedule[iteration]
    # else:
    #     for param_group in optimizer.param_groups:
    #         param_group["lr"] = lr_schedule

    # #make image list and labels tensor
    # # images = images.cuda(non_blocking=True)
    # # labels = labels.cuda(non_blocking=True)

    # # SGD
    # # optimizer.zero_grad()
    # # loss.backward()
    # # optimizer.step()

    # optimizer.zero_grad()

    # image_list = image_list.cuda(non_blocking=True)

    # with torch.cuda.amp.autocast(enabled=args.use_fp16):
    #     # without swav aug
    #     output = model(images)

    #     loss = criterion(output, labels)

    # # with swav aug
    # #     compute loss
    #     # low_resolution_output = model(low_resolution_batch)
    #     # high_resolution_output = model(high_resolution_batch)
        
    #     # output = torch.cat((high_resolution_output, low_resolution_output), dim=0)

    #     # loss = criterion(output, labels)

    # # update metric
    # losses.update(loss.item(), bsz)
    # acc1, acc5 = accuracy(output[:bsz], labels[:bsz], topk=(1, 5))
    # top1.update(acc1[0], bsz)

    # # SGD
    # # optimizer.zero_grad()
    # # loss.backward()
    # # optimizer.step()
    # # TODO: adjust when scaler is None
    # scaler.scale(loss).backward()
    # scaler.step(optimizer)
    # scaler.update()

    # # measure elapsed time
    # batch_time.update(time.time() - end)
    # end = time.time()

    # # print info
    # print('Train: [{0}][{1}/{2}]\t'
    #         'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #         'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
    #         'loss {loss.val:.3f} ({loss.avg:.3f})\t'
    #         'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
    #         epoch, idx + 1, len(train_loader), batch_time=batch_time,
    #         data_time=data_time, loss=losses, top1=top1))
            
    # print("weight mean", model.module.features[0][0].weight.mean(), 
    #         "weight std", model.module.features[0][0].weight.std())
    # sys.stdout.flush()

    for idx, (image_list, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc='epoch'):
        data_time.update(time.time() - end)

        # # Sepearte two batches for 2 high resolution crops, and the rest 6 for low resolution crops
        # high_resolution_batch = image_list[:2]
        # high_resolution_batch = torch.cat(high_resolution_batch, dim=0)
        # high_resolution_batch.cuda(non_blocking=True)
        # high_resolution_labels = labels.repeat(2)

        # # #low resultion batch is the rest in image_list all extended, and should be tensor
        # low_resolution_batch = image_list[2:]
        # low_resolution_batch = torch.cat(low_resolution_batch, dim=0)
        # low_resolution_batch.cuda(non_blocking=True)
        # low_resolution_labels = labels.repeat(len(image_list)-2)

        # labels = torch.cat((high_resolution_labels, low_resolution_labels), dim=0)

        labels = labels.cuda(non_blocking=True)

        bsz = labels.shape[0]

        # update learning rate
        iteration = epoch * len(train_loader) + idx
        if args.lr_scheduler:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[iteration]
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.use_fp16):
            # without swav aug
            # image_list = image_list.cuda(non_blocking=True)
            # output = model(image_list)

        # with swav aug
        #     compute loss
            # low_resolution_output = model(low_resolution_batch)
            # high_resolution_output = model(high_resolution_batch)
            # output = torch.cat((high_resolution_output, low_resolution_output), dim=0)
            output = model(image_list)
            loss = criterion(output, labels)

        if args.rank == 0 and args.wandb:
            wandb.log({'loss': loss.item()})

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output[:bsz], labels[:bsz], topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # TODO: adjust when scaler is None
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if args.rank == 0 and idx % 10 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

            if args.wandb:
                wandb.log({'conv2d weight mean': model.module.features[0][0].weight.mean(),
                'conv2d weight std': model.module.features[0][0].weight.std()})

            print("weight mean", model.module.features[0][0].weight.mean(), 
            "weight std", model.module.features[0][0].weight.std())
        
    return (epoch, losses.avg), losses.avg, top1.avg

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == "__main__":
    set_seed(42)
    wandb.require("service")
    main()
