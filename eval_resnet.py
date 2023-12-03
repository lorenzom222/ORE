import os
import sys
import argparse
import time
import math
from tqdm import tqdm
import wandb
import random
import csv
import numpy as np

import torch
import torch.nn as nn
from torchvision import datasets, transforms
# from src.mobilenet_modified_gelu_ER import MobNet_StartAt_Layer8
from torchvision.models import resnet18
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# from src.mobilenet_modified_gelu2 import MobNet_StartAt_Layer8

from src.utils import AverageMeter, accuracy, bool_flag, set_seed, average_precision_score

'''python3 eval_linear_mobilenet.py --epochs 100 --ckpt_path /home/nina/swav/experiments/MobNet_Large_Wider_ore_african_skew_sl_aug/checkpoint.pth.tar --num_classes 40 --pretrain_classes 5940 --data_path /home/nina/eval_dataset --wandb False --batch_size 64 --track_race_acc True'''

'''python3 eval_linear_mobilenet.py --epochs 50 --ckpt_path /home/nina/swav/experiments/MobNet_Large_Wider_ore_indian_skew_sl_aug/checkpoint.pth.tar --num_classes 40 --pretrain_classes 5940 --data_path /home/nina/eval_dataset --wandb True --batch_size 64 --track_race_acc True'''

'''python3 eval_linear_mobilenet.py --epochs 50 --ckpt_path /home/nina/swav/experiments/MobNet_Large_Wider_ore_balanced_no_aug/checkpoint.pth.tar --num_classes 1960 --pretrain_classes 5940 --data_path /home/nina/ore_balanced_dataset --wandb False --batch_size 64 --track_race_acc True'''

'''python3 eval_linear_mobilenet.py --epochs 50 --ckpt_path /home/nina/swav/experiments/MobNet_Large_Wider_ore_balanced_mini_aug_sl/checkpoint.pth.tar --num_classes 1960 --pretrain_classes 40 --data_path /home/nina/ore_balanced_dataset --wandb False --batch_size 64 --track_race_acc True'''

'''python3 eval_linear_mobilenet.py --epochs 50 --ckpt_path /home/nina/swav/experiments/MobNet_Large_Wider_ore_balanced_mini_no_aug_sl/checkpoint.pth.tar --num_classes 1960 --pretrain_classes 200 --data_path /home/nina/eval_dataset --wandb False --batch_size 64 --track_race_acc True'''

'''python3 linear_eval.py --epochs 10 --ckpts /home/nina/resnet/experiments/ResNet_ore_balanced_mini_aug_sl/checkpoints --num_classes 40 --pretrain_classes 200 --data_path /home/nina/eval_dataset --wandb False --batch_size 64 --track_race_acc True'''

'''python3 linear_eval.py --epochs 10 --ckpts /home/nina/resnet/experiments/ResNet_ore_balanced_mini_no_aug_sl/checkpoints --num_classes 40 --pretrain_classes 200 --data_path /home/nina/eval_dataset --wandb False --batch_size 64 --track_race_acc True'''

'''python3 linear_eval.py --epochs 30 --ckpts /home/nina/resnet/experiments/ResNet_ore_balanced_mini_no_aug_sl/checkpoints --num_classes 40 --pretrain_classes 200 --data_path /home/nina/eval_dataset --wandb False --batch_size 64 --track_race_acc True'''

'''python3 linear_eval.py --epochs 30 --ckpt_path /home/nina/resnet/experiments/ResNet_ore_balanced_mini_aug_sl/checkpoint.pth.tar --num_classes 40 --pretrain_classes 200 --data_path /home/nina/eval_dataset --wandb False --batch_size 64 --track_race_acc True'''

'''python3 linear_eval_ore.py --epochs 30 --ckpt_path /home/nina/resnet/experiments/ResNet_african_skew_no_aug_sl/checkpoint.pth.tar --num_classes 40 --pretrain_classes 5938 --data_path /home/nina/eval_dataset --wandb False --batch_size 64 --track_race_acc True'''


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    
    #model
    parser.add_argument('--ckpt_path', type=str, help='path to checkpoint')
    parser.add_argument('--data_path', type=str, help='path to data folder')
    parser.add_argument('--num_classes', type=int, default=1960, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--epochs', type=int, default=40, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--print_freq', type=int, default=10,
                    help='print frequency')
    parser.add_argument('--wandb', default=False, type=bool_flag, help='use wandb')
    parser.add_argument('--track_race_acc', default=False, type=bool_flag, help='track accuracy by race')
    parser.add_argument('--ckpts', nargs='+', help='list of ckpts to evaluate')

    
    #optimization
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--pretrain_classes', type=int, default=1960)
    parser.add_argument('--early_stopping', default=False, type=bool_flag, help='use early stopping')
    parser.add_argument('--run_name', type=str, help='WANDB run name')
    parser.add_argument('--save', type=str, help='Save metrics to npz file')

    opt = parser.parse_args()

    if opt.data_path is None:
        raise ValueError('one or more of the folders is None: data_folder')
    else:
        opt.train_folder = os.path.join(opt.data_path, 'train')
        opt.val_folder = os.path.join(opt.data_path, 'test')
    
    if opt.ckpt_path is None:
        raise ValueError('one or more of the folders is None: ckpt_path')
    
    if opt.num_classes is None:
        raise ValueError('one or more of the folders is None: num_classes')
    
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt

def set_loader(opt):
    #get train and val loader
    input_shape = [3, 224, 224]

    #imagenet mean and std
    mean = [0.485, 0.456, 0.406]
    std = [0.228, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomCrop(input_shape[1:]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                        std=std)
    ])  

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                        std=std)
    ])

    if opt.train_folder is not None:
        train_dataset = datasets.ImageFolder(
            root=opt.train_folder,
            transform=train_transform
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=True,
            num_workers=opt.num_workers, pin_memory=True)
    else:
        train_loader = None

    val_dataset = datasets.ImageFolder(
        root=opt.val_folder,
        transform=val_transform
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader, val_loader

def set_model(opt, ckpt_path=None):

    # Load the pre-trained ResNet18 model
    net = resnet18(pretrained=False, num_classes=opt.pretrain_classes).to('cuda')

    # Load your checkpoint
    checkpoint = torch.load(opt.ckpt_path)

    new_state_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        new_key = key.replace('module.', '')  # Remove the 'module.' prefix
        new_state_dict[new_key] = value

    net.load_state_dict(new_state_dict)

    # Freeze all layers except the final classification layer
    # for param in net.parameters():
    #     param.requires_grad = False
    
    # # Replace the final classification layer
    # net.fc = nn.Linear(net.fc.in_features, opt.num_classes)
    # net.fc.weight.data.normal_(mean=0.0, std=0.01)
    # net.fc.bias.data.zero_()

    # Set criterion
    criterion = nn.CrossEntropyLoss().cuda()

    return net.to('cuda:0'), criterion

def set_optimizer(opt, net):

    # Collect parameters
    optimizer = torch.optim.SGD(net.fc.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
    return optimizer

def train(train_loader, net, criterion, optimizer, epoch, opt):
    """one epoch training"""
    net.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Train epoch= %d' % epoch):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = net(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        # ap_score.update(average_precision_score(labels.cpu().detach().numpy(), output.cpu().detach().numpy()), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def label_to_race(labels, id_to_idx, id_to_race):
    id_list = labels.tolist() 
  
    class_labels = [class_label for idx in id_list for class_label, class_idx in id_to_idx.items() if class_idx == idx] # Get the list of corresponding ids that the images are from
    # id_list: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    # -> Class: ['m.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.01lb8z', 'm.01lb8z', 'm.01lb8z', 'm.01lb8z']
    races = [id_to_race[class_label] for class_label in class_labels] # Now map the ids to the race from the csv
    # print(f"Class: {class_labels}")
    # print(f"Races: {races}")
    # Create a dictionary mapping labels to their corresponding races
    label_to_race_mapping = dict(zip(id_list, races))
    # print(f"Label-to-Race{label_to_race_mapping}")

    return label_to_race_mapping

def validate(val_loader, net, criterion, opt):
    """validation"""
    net.eval()
    id_to_idx = val_loader.dataset.class_to_idx # map of id to indices in the val set

    if opt.track_race_acc:
        caucasian_ap = AverageMeter()
        indian_ap = AverageMeter()
        asian_ap = AverageMeter()
        african_ap = AverageMeter()

        caucasian_am = AverageMeter()
        indian_am = AverageMeter()
        asian_am = AverageMeter()
        african_am = AverageMeter()
        # Initialize AverageMeter instances for AUC
        caucasian_auc = AverageMeter()
        indian_auc = AverageMeter()
        asian_auc = AverageMeter()
        african_auc = AverageMeter()

        caucasian_fpr = AverageMeter()
        indian_fpr = AverageMeter()
        asian_fpr = AverageMeter()
        african_fpr = AverageMeter()

        caucasian_tpr = AverageMeter()
        indian_tpr = AverageMeter()
        asian_tpr = AverageMeter()
        african_tpr = AverageMeter()

        race_to_ap = {'Caucasian': caucasian_ap, 'Indian': indian_ap, 'Asian': asian_ap, 'African': african_ap}
        race_acc = {
            0: ("Caucasian", caucasian_am),
            1: ("Indian", indian_am),
            2: ("Asian", asian_am),
            3: ("African", african_am),
        }
        # Dictionary to map races to their respective AUC AverageMeters
        race_to_auc = {'Caucasian': caucasian_auc, 'Indian': indian_auc, 'Asian': asian_auc, 'African': african_auc}
        race_to_fpr = {'Caucasian': [], 'Indian': [], 'Asian': [], 'African': []}
        race_to_tpr = {'Caucasian': [], 'Indian': [], 'Asian': [], 'African': []}

        race_name = {
            "Caucasian": '0',
            "Indian": '1',
            "Asian": '2',
            "African": '3'
        }
        id_to_race = {} 
        with open('/home/lorenzo/ore-dir/swav/data/experiments/drive/final_data/linear_prob_dataset_split.csv', 'r') as csvfile:
        # with open('/home/nina/SupContrast-ORE/data/new_balanced_data_split_final.csv', 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                id_to_race[row['id']] = row['race']
    else:
        race_to_ap = None

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    all_labels = []
    all_outputs = []

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            #get race accuracy
            labels_to_races = label_to_race(labels, id_to_idx, id_to_race)

            # forward
            output = net(images)            
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)

            # if opt.track_race_acc:
            #     acc1, accuracy_by_race = accuracy(output, labels, topk=(1,5), label_to_race_mapping = labels_to_races)
            #     accuracy_by_race_1 = accuracy_by_race[0]
            #     accuracy_by_race_5 = accuracy_by_race[1]
            #     top1.update(acc1[0][0], bsz)

            if opt.track_race_acc:
                #get average precision
                # race_ap_batch = average_precision_score(y_true=labels.cpu().detach().numpy(), y_score=output.cpu().detach().numpy(), labels_to_races=labels_to_races)
                # for race in ['Caucasian', 'Indian', 'African', 'Asian']:
                #     if race_ap_batch[race][1] > 0:
                #         race_to_ap[race].update(race_ap_batch[race][0], race_ap_batch[race][1])
                #get accuracy
                acc1, accuracy_by_race = accuracy(output, labels, topk=(1,5), label_to_group_mapping = labels_to_races)
                accuracy_by_race_1 = accuracy_by_race[0]
                top1.update(acc1[0][0], bsz)

                for race, acc in accuracy_by_race_1.items():
                    race = int(race)  # Convert race to an integer
                    name = race_acc[race][0]
                    meter = race_acc[race][1]
                    meter.update(acc, bsz)
            else:
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #accumate labels and outputs
            all_labels.append(labels.cpu().detach().numpy())
            all_outputs.append(output.cpu().detach().numpy())

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
                # if opt.track_race_acc:
                    # for name, meter in race_to_ap.items():
                    #     print(f'{name} AP {meter.val:.3f} ({meter.avg:.3f})')
                    # print('\nLinear prob acc')
                    # for race, (name, meter) in race_acc.items():
                    #     print(f'{name} acc {meter.val:.3f} ({meter.avg:.3f})')
    
    all_labels = np.concatenate(all_labels, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)

    #calculate ap
    labels_to_races = label_to_race(all_labels, id_to_idx, id_to_race)
    total_race_ap = average_precision_score(y_score=all_outputs, y_true=all_labels, labels_to_races=labels_to_races)
    # print(total_race_ap)

    for race in ['Caucasian', 'Indian', 'African', 'Asian']:
        race_key = race_name[race]
        if total_race_ap[race_key][1] > 0:
            race_to_ap[race].update(total_race_ap[race_key][0], total_race_ap[race_key][1])

    # Calculate AUC for each race and update the AverageMeters
    for race in ['Caucasian', 'Indian', 'African', 'Asian']:
        race_key = race_name[race]
        
        # Create a boolean array indicating which samples belong to the current race
        race_labels = np.array([labels_to_races[idx] == race_key for idx in all_labels])
        
        # Select the corresponding scores for the current race
        race_outputs = all_outputs[:, int(race_key)]
        
        # Calculate the AUC score if there are any samples for the race
        if race_labels.any():
            auc_score = roc_auc_score(race_labels, race_outputs)
            race_to_auc[race].update(auc_score)
    # Calculate FPR for each race
    # fpr_dict = {}
    for race in ['Caucasian', 'Indian', 'African', 'Asian']:
        race_key = race_name[race]
        
        # Create a boolean array indicating which samples belong to the current race
        race_labels = np.array([labels_to_races[idx] == race_key for idx in all_labels])
        
        # Select the corresponding scores for the current race
        race_outputs = all_outputs[:, int(race_key)]
        
        # Calculate the ROC curve if there are any samples for the race
        if race_labels.any():
            fpr, tpr, thresholds = roc_curve(race_labels, race_outputs)
            race_to_fpr[race].append(fpr)
            race_to_tpr[race].append(tpr)


    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1)) 

    return losses.avg, top1.avg, race_to_ap, race_acc, race_to_auc, race_to_fpr, race_to_tpr

def main():
    best_acc = {'Total': 0, 'Caucasian': 0, 'Indian': 0, 'Asian': 0, 'African': 0}
    patience = 5
    best_race_ap = None
    best_race_acc = None
    delta = 0.001
    opt = parse_option()

    if opt.wandb:
        wandb.init(project='eval_resnet', entity='ore-dreamteam', group=f'eval_{opt.run_name}_{opt.epochs}')

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # for ckpt in os.listdir(opt.ckpts[0]):
        
    #     print("\n======================{}=========================\n".format(ckpt))

    #     ckpt_path = os.path.join(opt.ckpts[0], ckpt)

    # build model and criterion
    # net, criterion = set_model(opt, ckpt_path=ckpt_path)
    net, criterion = set_model(opt)


    # build optimizer
    optimizer = set_optimizer(opt, net)

    val_loss, val_acc, race_to_ap, race_acc, race_to_auc,race_to_fpr,race_to_tpr = validate(val_loader, net, criterion, opt)


    # training routine
    for epoch in range(1, opt.epochs + 1):
        #not adjust learning rate for now

        # # train for one epoch
        time1 = time.time()
        train_loss, train_acc = train(train_loader, net, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        print('Train Loss: {:.4f}, Train Acc: {:.4f}'.format(train_loss, train_acc))

        # eval for one epoch
        val_loss, val_acc, race_to_ap, race_acc, race_to_auc,race_to_fpr,race_to_tpr = validate(val_loader, net, criterion, opt)
        # Log and print the average precision (AP) for each race
        print("Average Precision (AP) for each race:")
        for race, meter in race_to_ap.items():
            print(f"{race} AP: {meter.avg:.3f} (current value: {meter.val:.3f})")

        # Log and print the accuracy for each race
        print("\nAccuracy for each race:")
        for race, (name, meter) in race_acc.items():
            print(f"{name} accuracy: {meter.avg:.3f} (current value: {meter.val:.3f})")

        # Log and print the AUC for each race
        # Print out the AUC scores using the AverageMeter's .avg attribute
        print("\nArea Under the Curve (AUC) for each race:")
        for race, meter in race_to_auc.items():
            print(f"{race} AUC: {meter.avg:.3f} (current value: {meter.val:.3f})")

        # print(f"\nFPR: {race_to_fpr}")
        # for race, array in race_to_fpr.items():
        #     print(f"{race} AUC: {meter.avg:.3f} (current value: {meter.val:.3f})")
        # print(f"\nTPR: {race_to_tpr}")
        # for race, array in race_to_tpr.items():
        #     print(f"{race} AUC: {meter.avg:.3f} (current value: {meter.val:.3f})")

        if opt.wandb:
            wandb.log({
                'epoch': epoch,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                })
            for race, (name, meter) in race_acc.items():
                wandb.log({
                    "Accuracy plot": {
                        f'{name}_accuracy_avg': meter.avg,
                        f'{name}_accuracy_val': meter.val
                    }
                }, step=epoch)
            for race, meter in race_to_auc.items():
                wandb.log({
                    "AUC plot": {
                        f'{race}_AUC_avg': meter.avg,
                        f'{race}_AUC_val': meter.val
                    }
                }, step=epoch)
            for race, meter in race_to_ap.items():
                wandb.log({
                    "AP plot": {
                        f'{race}_AP_avg': meter.avg,
                        f'{race}_AP_val': meter.val
                    }
                }, step=epoch)

        if opt.save:
            # Create the "metrics" directory if it does not exist
            if not os.path.exists('metrics'):
                os.makedirs('metrics')

            # Define the filename for the .npz file, including the path to the "metrics" folder
            filename = os.path.join('metrics', f'metrics_{opt.run_name}.npz')
            print(f"Saving metrics to {filename}...")

            # Load the existing data if the file already exists
            if os.path.exists(filename):
                data = np.load(filename, allow_pickle=True)
                metrics_dict = dict(data)
                data.close()
            else:
                metrics_dict = {}

            current_epoch_metrics = {
                'val_loss': val_loss.cpu().numpy() if isinstance(val_loss, torch.Tensor) else val_loss,
                'train_loss': train_loss.cpu().numpy() if isinstance(train_loss, torch.Tensor) else train_loss,
                'train_acc': train_acc.cpu().numpy() if isinstance(train_acc, torch.Tensor) else train_acc,
                'val_acc': val_acc.cpu().numpy() if isinstance(val_acc, torch.Tensor) else val_acc,
                'AP': {
                    race: {
                        'avg': meter.avg.cpu().numpy() if isinstance(meter.avg, torch.Tensor) else meter.avg,
                        'val': meter.val.cpu().numpy() if isinstance(meter.val, torch.Tensor) else meter.val
                    } for race, meter in race_to_ap.items()
                },
                'Accuracy': {
                    name: {
                        'avg': meter.avg.cpu().numpy() if isinstance(meter.avg, torch.Tensor) else meter.avg,
                        'val': meter.val.cpu().numpy() if isinstance(meter.val, torch.Tensor) else meter.val
                    } for race, (name, meter) in race_acc.items()
                },
                'AUC': {
                    race: {
                        'avg': meter.avg.cpu().numpy() if isinstance(meter.avg, torch.Tensor) else meter.avg,
                        'val': meter.val.cpu().numpy() if isinstance(meter.val, torch.Tensor) else meter.val
                    } for race, meter in race_to_auc.items()
                },
                'FPR': {
                    race: {
                        'array': array
                    } for race, array in race_to_fpr.items()
                },
                'TPR': {
                    race: {
                        'array': array
                    } for race, array in race_to_tpr.items()
                }

            }

            # Add the current epoch's metrics to the metrics_dict using the epoch number as the key
            metrics_dict[f'epoch_{epoch}'] = current_epoch_metrics

            # Save the updated metrics_dict to the .npz file in the "metrics" folder
            np.savez(filename, **metrics_dict)

        #add weighted improvement
        delta_weights = {'Total': 0.5, 'Caucasian': 0.125, 'Indian': 0.125, 'Asian': 0.125, 'African': 0.125}
        weighted_improvement = (delta_weights['Total'] * (val_acc - best_acc['Total']) +
                        delta_weights['Caucasian'] * (race_acc[0][1].avg - best_acc['Caucasian']) +
                        delta_weights['Indian'] * (race_acc[1][1].avg - best_acc['Indian']) +
                        delta_weights['Asian'] * (race_acc[2][1].avg - best_acc['Asian']) +
                        delta_weights['African'] * (race_acc[3][1].avg - best_acc['African']))
        if weighted_improvement > delta:
            best_acc['Total'] = val_acc
            best_acc['Caucasian'] = race_acc[0][1].avg
            best_acc['Indian'] = race_acc[1][1].avg
            best_acc['Asian'] = race_acc[2][1].avg
            best_acc['African'] = race_acc[3][1].avg

            best_epoch = epoch
            patience = 5
            best_race_ap = race_to_ap
        else:
            patience -= 1


            
        print('Val Loss: {:.4f}, Val Acc: {:.4f}'.format(val_loss, val_acc))

        if opt.early_stopping and patience==0:
            print("Stopping early...")
            break
        #not saving model yet

    print('best epoch', best_epoch)
    print('best total accuracy: {:.2f}'.format(best_acc['Total']))
    if opt.track_race_acc:
        print('best acc and ap')
        for name, meter in best_race_ap.items():
                print(f'{name} mean AP ({meter.avg:.3f})')
        print('\nLinear prob acc')
        # for _, (name, meter) in best_race_acc.items():
        #     print(f'{name} acc {meter.val:.3f} ({meter.avg:.3f})')
        for name, acc in best_acc.items():
            print(f'{name} acc {acc:.3f}')

        print("\ncur_epoch{}, cur stats: ".format(epoch))
        for name, meter in race_to_ap.items():
                print(f'{name} mean AP ({meter.avg:.3f})')
        print('\nLinear prob acc')
        for _, (name, meter) in race_acc.items():
            print(f'{name} acc {meter.val:.3f} ({meter.avg:.3f})')

if __name__ == '__main__':
    set_seed(42)
    wandb.require('service')

    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    wandb.finish()