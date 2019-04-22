import argparse
import os
import sys
import time
import torch

from torch.autograd import Variable
from torch import nn
from torch import optim
from torch.nn import init
from torch.backends import cudnn
import numpy as np

from data import SSDAugmentation, Dataset, detection_collate
from config import synme, DATASET_ROOT, MEANS
from utils import str2bool, load_weights, save_weights
from layers.modules import MultiBoxLoss
from ssd import build_ssd

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset_root', default=DATASET_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    cfg = synme
    dataset = Dataset(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))

    net = build_ssd('train', synme) 

    if args.resume:
        print('Resume training...')
        load_weights(net, args.resume)
    else:
        vgg_weights = args.save_folder + args.basenet
        print('Loading base network...')
        load_weights(net.vgg, vgg_weights)

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        net.extras.apply(weights_init)
        net.loc.apply(weights_init)
        net.conf.apply(weights_init)

    if args.cuda:
        net = torch.nn.DataParallel(net).cuda()
        cudnn.benchmark = True


    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, synme['lr_steps'])
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)


    data_loader = torch.utils.data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if iteration != 0 and (iteration % epoch_size == 0):
            epoch += 1

        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), requires_grad=False) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, requires_grad=False) for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        t1 = time.time()


        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data.item()), end=' ')


        if iteration != 0 and iteration % 500 == 0:
            print('Saving state, iter:', iteration)
            pkl_file = 'ssd300_synme_' + repr(iteration) + '.pth'
            save_weights(net, os.path.join(args.save_folder, pkl_file))

    pkl_file = args.save_folder + dataset.name + '.pth'
    save_weights(net, os.path.join(args.save_folder, pkl_file))


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight.data)
        m.bias.data.zero_()


if __name__ == '__main__':
    train()
