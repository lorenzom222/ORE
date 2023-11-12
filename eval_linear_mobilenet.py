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
from src.mobilenet_modified_gelu2 import MobNet_StartAt_Layer8

from src.utils import AverageMeter, accuracy, bool_flag, set_seed, average_precision_score

'''python3 eval_linear_mobilenet.py --epochs 100 --ckpt_path /home/nina/swav/experiments/MobNet_Large_Wider_ore_african_skew_sl_aug/checkpoint.pth.tar --num_classes 40 --pretrain_classes 5940 --data_path /home/nina/eval_dataset --wandb False --batch_size 64 --track_race_acc True'''

'''python3 eval_linear_mobilenet.py --epochs 50 --ckpt_path /home/nina/swav/experiments/MobNet_Large_Wider_ore_indian_skew_sl_aug/checkpoint.pth.tar --num_classes 40 --pretrain_classes 5940 --data_path /home/nina/eval_dataset --wandb False --batch_size 64 --track_race_acc True'''

'''python3 eval_linear_mobilenet.py --epochs 50 --ckpt_path /home/nina/swav/experiments/MobNet_Large_Wider_ore_indian_skew_sl_aug/checkpoint.pth.tar --num_classes 1960 --pretrain_classes 5940 --data_path /home/nina/ore_balanced_dataset --wandb False --batch_size 64 --track_race_acc True'''


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
    parser.add_argument('--track_supcat_acc', default=False, type=bool_flag, help='track accuracy by super categories (INAT)')

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
    parser.add_argument('--full_folder', type=str, help='path to checkpoint')

    opt = parser.parse_args()

    if opt.data_path is None:
        if opt.full_folder is None:
            raise ValueError('one or more of the folders is None: data_folder')
        opt.train_folder = None
        opt.val_folder = None
    

        # raise ValueError('one or more of the folders is None: data_folder')
    else:
        opt.train_folder = os.path.join(opt.data_path, 'train')
        opt.val_folder = os.path.join(opt.data_path, 'val')
        opt.full_folder = None
    

    if opt.ckpt_path is None:
        raise ValueError('one or more of the folders is None: ckpt_path')
    
    if opt.num_classes is None:
        raise ValueError('one or more of the folders is None: num_classes')
    
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt

from torch.utils.data import random_split

def set_loader(opt):
    #get train and val loader
    input_shape = [3, 224, 224]
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(input_shape[1:]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
    ])  

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
    ])

    if opt.train_folder is not None:
        train_dataset = datasets.ImageFolder(
            root=opt.train_folder,
            transform=train_transform
        )
    elif opt.full_folder is not None:
        full_dataset = datasets.ImageFolder(
            root=opt.full_folder
        )
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform
    else:
        train_dataset = None

    if train_dataset is not None:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=True,
            num_workers=opt.num_workers, pin_memory=True)

    if opt.val_folder is not None:
        val_dataset = datasets.ImageFolder(
            root=opt.val_folder,
            transform=val_transform
        )
    elif opt.full_folder is not None and train_dataset is not None:
        pass  # val_dataset is already defined
    else:
        val_dataset = None

    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True)
    else:
        val_loader = None

    return train_loader, val_loader



def set_model(opt):

    checkpoint = torch.load(opt.ckpt_path)
    net = eval('MobNet_StartAt_Layer8')(num_classes=opt.num_classes, pretrain_classes=opt.pretrain_classes).to('cuda')

    # for param_tensor in net.state_dict():
    #     print(param_tensor, "\t", net.state_dict()[param_tensor].size())
    # print(net)

    net = torch.nn.DataParallel(net).to('cuda:0')

    #replace module from keys with model.model.
    for key in list(checkpoint['state_dict'].keys()):
        if key.startswith('module.'):
            checkpoint['state_dict'][key.replace('module.', 'module.model.')] = checkpoint['state_dict'].pop(key)

    net.load_state_dict(checkpoint['state_dict'])

    #don't need to remove layers
    # for _ in range(0, 8):
    #     del net.module.model.features[0]
    

    #freeze feature layer
    # for param_tensor in net.state_dict():
    #     if not param_tensor.startswith('model.classifier.3'):
    #         net.state_dict()[param_tensor].requires_grad = False
    net.requires_grad_(False)
    net.module.model.classifier[3].requires_grad_(True)

    net.module.model.classifier[3] = torch.nn.Linear(1280, opt.num_classes) 
    net.module.model.classifier[3].weight.data.normal_(mean=0.0, std=0.01)
    net.module.model.classifier[3].bias.data.zero_()

    #set criterion
    criterion = nn.CrossEntropyLoss().cuda()

    return net.to('cuda:0'), criterion

def set_optimizer(opt, net):

    # Collect parameters
    classifier_parameters = []
    frozen_parameters = []
    for k, param in net.named_parameters():
        if "classifier.3" in k:
            classifier_parameters.append(param)
        else:
            frozen_parameters.append(param)

    # Create optimizer  
    param_groups = [
        {'params': classifier_parameters, 'lr': opt.learning_rate},
        {'params': frozen_parameters, 'lr': 0},
    ]
    optimizer = torch.optim.SGD(param_groups) 

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

    races = [id_to_race[class_label] for class_label in class_labels] 
    label_to_race_mapping = dict(zip(id_list, races))

    return label_to_race_mapping

def label_to_supcat(labels, id_to_idx, id_to_supcat):
    id_list = labels.tolist() 
  
    class_labels = [class_label for idx in id_list for class_label, class_idx in id_to_idx.items() if class_idx == idx] # Get the list of corresponding ids that the images are from


    label_to_sup_mapping = dict(zip(id_list, class_labels))

    return label_to_sup_mapping

def set_meters(opt, val_loader):     
    if opt.track_race_acc:
        class_to_idx = val_loader.dataset.class_to_idx
    elif opt.track_supcat_acc:
        class_to_idx = val_loader.dataset.dataset.class_to_idx


    # id_to_race, id_to_sup = None, None
    # race_to_ap, sup_to_ap = None, None
    res = None

    if opt.track_race_acc:
        race_to_ap = {race: AverageMeter() for race in ['Caucasian', 'Indian', 'Asian', 'African']}
        race_acc = {i: (race, AverageMeter()) for i, race in enumerate(race_to_ap.keys())}
        id_to_race = {row['id']: row['race'] for row in csv.DictReader(open('/home/nina/SupContrast-ORE/data/new_balanced_data_split_final.csv', 'r'))}
        res =  (race_to_ap, race_acc, id_to_race)
    elif opt.track_supcat_acc:
        sup_to_ap = {sup: AverageMeter() for sup in ['Birds', 'Insects', 'Plants']}
        # print(sup_to_ap)
        sup_acc = {sup: AverageMeter() for sup in sup_to_ap.keys()}
        id_to_sup = {row['CategoryID']: row['SuperCategory'] for row in csv.DictReader(open('/home/lorenzo/csv_files_mini_experiments/linear_prob_dataset.csv', 'r'))}
        res =  (sup_to_ap, sup_acc, id_to_sup)

    return class_to_idx, res


def validate(val_loader, net, criterion, opt):
    """validation"""
    net.eval()
    id_to_idx, (superclass_to_ap, superclass_to_acc, id_to_superclass) = set_meters(opt, val_loader)


    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    all_labels = []
    all_outputs = []
    labels_to_superclass = None

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            #get race accuracy
            if opt.track_race_acc:
                labels_to_superclass = label_to_race(labels, id_to_idx, id_to_superclass)
            elif opt.track_supcat_acc:
                labels_to_superclass = label_to_supcat(labels, id_to_idx, id_to_superclass)

            # forward
            output = net(images)            
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)


            if opt.track_race_acc:

                acc1, accuracy_by_race = accuracy(output, labels, topk=(1,5), label_to_race_mapping = labels_to_superclass)
                accuracy_by_race_1 = accuracy_by_race[0]
                top1.update(acc1[0][0], bsz)

                for race, acc in accuracy_by_race_1.items():
                    race = int(race)  # Convert race to an integer
                    name = superclass_to_acc[race][0]
                    meter = superclass_to_acc[race][1]
                    meter.update(acc, bsz)

            elif opt.track_supcat_acc:

                acc1, accuracy_by_supcat = accuracy(output, labels, topk=(1,5), label_to_race_mapping = labels_to_superclass)
                accuracy_by_supcat_1 = accuracy_by_supcat[0]
                top1.update(acc1[0][0], bsz)

                for supcat, acc in accuracy_by_supcat_1.items():
                    name = superclass_to_acc[supcat]
                    meter = superclass_to_acc[supcat]
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

    all_labels = np.concatenate(all_labels, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)

    if opt.track_race_acc:
        labels_to_superclass = label_to_race(labels, id_to_idx, id_to_superclass)
    elif opt.track_supcat_acc:
        labels_to_superclass = label_to_supcat(labels, id_to_idx, id_to_superclass)

    total_superclass_ap = average_precision_score(y_score=all_outputs, y_true=all_labels, labels_to_races=labels_to_superclass)

    for superclass in list(superclass_to_ap.keys()):
        if total_superclass_ap[superclass][1] > 0:
            superclass_to_ap[superclass].update(total_superclass_ap[superclass][0], total_superclass_ap[superclass][1])

    for name, meter in superclass_to_ap.items():
        print(f'{name} AP {meter.val:.3f} ({meter.avg:.3f})')

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1)) 

    return losses.avg, top1.avg, superclass_to_ap, superclass_to_acc

def main():
    # best_acc = {'Total': 0, 'Caucasian': 0, 'Indian': 0, 'Asian': 0, 'African': 0}
    best_acc = {'Total': 0, 'Birds': 0, 'Plants': 0, 'Insects': 0}

    patience = 5
    best_race_ap = None
    best_race_acc = None
    delta = 0.001
    opt = parse_option()

    if opt.wandb:
        wandb.init(project='swav_mobilenet', entity='ore-dreamteam', group='eval_aug_{}_epochs'.format(opt.epochs))

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    net, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, net)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        #not adjust learning rate for now

        # # train for one epoch
        time1 = time.time()
        train_loss, train_acc = train(train_loader, net, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        print('Train Loss: {:.4f}, Train Acc: {:.4f}'.format(train_loss, train_acc))
         
        loss, val_acc, superclass_to_ap, superclass_acc = validate(val_loader, net, criterion, opt)

        # get the keys from superclass_acc
        superclasses = list(superclass_acc.keys())

        # make delta_weights = {'Total': 0.5, other keys: (distribution that all add up to 1.0)} 
        delta_weights = {superclass: 0.5 / len(superclasses) for superclass in superclasses}
        delta_weights['Total'] = 0.5

        # then update weighted_improvement from superclass_acc
        weighted_improvement = delta_weights['Total'] * (val_acc - best_acc['Total'])
        # print("superclass_acc: ", superclass_acc)
        for superclass in superclasses:
            weighted_improvement += delta_weights[superclass] * (superclass_acc[superclass].avg - best_acc[superclass])

        if weighted_improvement > delta:
            best_acc['Total'] = val_acc
            for superclass in superclasses:
                best_acc[superclass] = superclass_acc[superclass].avg
            best_epoch = epoch
            patience = 5
            best_sup_ap = superclass_to_ap
        else:
            patience -= 1
        if opt.wandb:
            wandb.log({'epoch': epoch, 
                       'val_loss': loss,
                       'train_loss': loss, 
                       'train_acc': train_acc,
                       'val_acc': val_acc,})
            
        print('Val Loss: {:.4f}, Val Acc: {:.4f}'.format(loss, val_acc))

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
        for name, meter in best_sup_ap.items():
                print(f'{name} mean AP ({meter.avg:.3f})')
        print('\nLinear prob acc')
        for _, (name, meter) in superclass_acc.items():
            print(f'{name} acc {meter.val:.3f} ({meter.avg:.3f})')

if __name__ == '__main__':
    set_seed(42)
    wandb.require('service')
    main()
