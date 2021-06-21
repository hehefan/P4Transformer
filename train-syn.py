from __future__ import print_function
import datetime
import os
import time
import sys
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import utils

from scheduler import WarmupMultiStepLR

from datasets.synthia import *
import models.synthia as Models


def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    for pc1, rgb1, label1, mask1, pc2, rgb2, label2, mask2 in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()

        pc1, rgb1, label1, mask1 = pc1.to(device), rgb1.to(device), label1.to(device), mask1.to(device)
        output1 = model(pc1, rgb1).transpose(1, 2)
        loss1 = criterion(output1, label1)*mask1
        loss1 = torch.sum(loss1) / (torch.sum(mask1) + 1)
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        pc2, rgb2, label2, mask2 = pc2.to(device), rgb2.to(device), label2.to(device), mask2.to(device)
        output2 = model(pc2, rgb2).transpose(1, 2)
        loss2 = criterion(output2, label2)*mask1
        loss2 = torch.sum(loss2) / (torch.sum(mask2) + 1)
        optimizer.zero_grad()
        loss2.backward()
        optimizer.step()

        metric_logger.update(loss=(loss1.item()+loss2.item())/2.0, lr=optimizer.param_groups[0]["lr"])
        lr_scheduler.step()
        sys.stdout.flush()

def evaluate(model, criterion, data_loader, device, print_freq):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_loss = 0
    total_correct = 0
    total_seen = 0
    total_pred_class = [0] * 12
    total_correct_class = [0] * 12
    total_class = [0] * 12

    with torch.no_grad():
        for pc1, rgb1, label1, mask1, pc2, rgb2, label2, mask2 in metric_logger.log_every(data_loader, print_freq, header):
            pc1, rgb1 = pc1.to(device), rgb1.to(device)
            output1 = model(pc1, rgb1).transpose(1, 2)
            loss1 = criterion(output1, label1.to(device))*mask1.to(device)
            loss1 = torch.sum(loss1) / (torch.sum(mask1.to(device)) + 1)
            label1, mask1 = label1.numpy().astype(np.int32), mask1.numpy().astype(np.int32)
            output1 = output1.cpu().numpy()
            pred1 = np.argmax(output1, 1) # BxTxN
            correct1 = np.sum((pred1 == label1) * mask1)
            total_correct += correct1
            total_seen += np.sum(mask1)
            for c in range(12):
                total_pred_class[c] += np.sum(((pred1==c) | (label1==c)) & mask1)
                total_correct_class[c] += np.sum((pred1==c) & (label1==c) & mask1)
                total_class[c] += np.sum((label1==c) & mask1)

            pc2, rgb2 = pc2.to(device), rgb2.to(device)
            output2 = model(pc2, rgb2).transpose(1, 2)
            loss2 = criterion(output2, label2.to(device))*mask2.to(device)
            loss2 = torch.sum(loss2) / (torch.sum(mask2.to(device)) + 1)
            label2, mask2 = label2.numpy().astype(np.int32), mask2.numpy().astype(np.int32)
            output2 = output2.cpu().numpy()
            pred2 = np.argmax(output2, 1) # BxTxN
            correct2 = np.sum((pred2 == label2) * mask2)
            total_correct += correct2
            total_seen += np.sum(mask2)
            for c in range(12):
                total_pred_class[c] += np.sum(((pred2==c) | (label2==c)) & mask2)
                total_correct_class[c] += np.sum((pred2==c) & (label2==c) & mask2)
                total_class[c] += np.sum((label2==c) & mask2)

            metric_logger.update(loss=(loss1.item()+loss2.item())/2.0)

    ACCs = []
    for c in range(12):
        acc = total_correct_class[c] / float(total_class[c])
        if total_class[c] == 0:
            acc = 0
        print('eval acc of %s:\t %f'%(index_to_class[label_to_index[c]], acc))
        ACCs.append(acc)
    print(' * Eval accuracy: %f'% (np.mean(np.array(ACCs))))

    IoUs = []
    for c in range(12):
        iou = total_correct_class[c] / float(total_pred_class[c])
        if total_pred_class[c] == 0:
            iou = 0
        print('eval mIoU of %s:\t %f'%(index_to_class[label_to_index[c]], iou))
        IoUs.append(iou)
    print(' * Eval mIoU:\t %f'%(np.mean(np.array(IoUs))))

def main(args):

    if args.output_dir:
        utils.mkdir(args.output_dir)

    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    # Data loading code
    print("Loading data")

    st = time.time()

    dataset = SegDataset(
            root=args.data_path,
            meta=args.data_train,
            labelweight=args.label_weight,
            frames_per_clip=args.clip_len,
            num_points=args.num_points,
            train=True
    )

    dataset_test = SegDataset(
            root=args.data_path,
            meta=args.data_eval,
            labelweight=args.label_weight,
            frames_per_clip=args.clip_len,
            num_points=args.num_points,
            train=False
    )

    print("Creating data loaders")

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    
    Model = getattr(Models, args.model)
    model = Model(radius=args.radius, nsamples=args.nsamples, num_classes=12)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion_train = nn.CrossEntropyLoss(weight=torch.from_numpy(dataset.labelweights).to(device), reduction='none')
    criterion_test = nn.CrossEntropyLoss(weight=torch.from_numpy(dataset_test.labelweights).to(device), reduction='none')

    lr = args.lr
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args.lr_warmup_epochs * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
        warmup_iters=warmup_iters, warmup_factor=1e-5)

    model_without_ddp = model

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion_train, optimizer, lr_scheduler, data_loader, device, epoch, args.print_freq)

        evaluate(model, criterion_test, data_loader_test, device=device, print_freq=args.print_freq)

        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            utils.save_on_master(
                checkpoint,
                os.path.join(output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(output_dir, 'checkpoint.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Transformer Model Training')

    parser.add_argument('--data-path', default='/scratch/HeheFan-data/Synthia4D/sequences', help='data path')
    parser.add_argument('--data-train', default='/scratch/HeheFan-data/Synthia4D/trainval_raw.txt', help='meta list for training')
    parser.add_argument('--data-eval', default='/scratch/HeheFan-data/Synthia4D/test_raw.txt', help='meta list for test')
    parser.add_argument('--label-weight', default='/scratch/HeheFan-data/Synthia4D/labelweights.npz', help='training label weights')
    
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model', default='P4Transformer', type=str, help='model')
    # input
    parser.add_argument('--clip-len', default=3, type=int, metavar='N', help='number of frames per clip')
    parser.add_argument('--num-points', default=16384, type=int, metavar='N', help='number of points per frame')
    # P4D
    parser.add_argument('--radius', default=0.9, type=float, help='radius for the ball query')
    parser.add_argument('--nsamples', default=32, type=int, help='number of neighbors for the ball query')
    parser.add_argument('--spatial-stride', default=16, type=int, help='spatial subsampling rate')
    parser.add_argument('--temporal-kernel-size', default=1, type=int, help='temporal kernel size')
    # embedding
    parser.add_argument('--emb-relu', default=False, action='store_true')
    # transformer
    parser.add_argument('--dim', default=1024, type=int, help='transformer dim')
    parser.add_argument('--depth', default=2, type=int, help='transformer depth')
    parser.add_argument('--head', default=4, type=int, help='transformer head')
    parser.add_argument('--mlp-dim', default=2048, type=int, help='transformer mlp dim')
    # training
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-milestones', nargs='+', default=[30, 40, 50], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=10, type=int, help='number of warmup epochs')
    # output
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='', type=str, help='path where to save')
    # resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
