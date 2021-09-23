import os 
import cv2
import sys
import tqdm
import time
import shutil
import logging
import argparse
import random
import numpy as np
import setproctitle
from skimage import measure
from skimage.color import label2rgb
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from models.denseunet import DenseUnet
from models.resunet import ResUNet
from models.unet import UNet
from dataset.abus_dataset_2d import ABUS_Dataset_2d, ElasticTransform, ToTensor, Normalize
from utils.loss import DiceLoss, SurfaceLoss, TILoss, MaskDiceLoss, MaskMSELoss
from utils.utils import save_checkpoint, confusion
from utils.lr_scheduler import LR_Scheduler

def get_args():
    print('------initing args------')
    parser = argparse.ArgumentParser()

    # general config
    parser.add_argument('--gpu', type=str, default='3')
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--seed', default=6, type=int) 
    parser.add_argument('--n_epochs', type=int, default=60)
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N')

    # dataset config
    parser.add_argument('--train_image_path', default='../semi-supervised/data/selected_data/image_100', type=str)
    parser.add_argument('--train_target_path', default='../semi-supervised/data/selected_data/label_100', type=str)
    parser.add_argument('--val_image_path', default='../semi-supervised/data/selected_data/val_image', type=str)
    parser.add_argument('--val_target_path', default='../semi-supervised/data/selected_data/val_label', type=str)
    parser.add_argument('--batchsize', type=int, default=5)

    # optimizer config
    parser.add_argument('--lr_scheduler', type=str, default='cos', choices=['poly', 'step', 'cos'])
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W')

    # network config
    parser.add_argument('--drop_rate', default=0.3, type=float)
    parser.add_argument('--arch', default='dense161', type=str, choices=('dense161', 'dense121', 'dense201', 'unet', 'resunet'))

    # frequently change args
    parser.add_argument('--log_dir', default='./log/super')
    parser.add_argument('--save', default='./work/super/test')

    args = parser.parse_args()
    return args


def main():
    #############
    # init args #
    #############
    args = get_args()

    # gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # creat save path
    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    # logger
    logging.basicConfig(filename=args.save+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info('--- init parameters ---')

    # writer
    idx = args.save.rfind('/')
    log_dir = args.log_dir + args.save[idx:]
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)

    # set title of the current process
    setproctitle.setproctitle(args.save)

    # random
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #####################
    # building  network #
    #####################
    logging.info('--- building network ---')

    if args.arch == 'dense121': 
        model = DenseUnet(arch='121', pretrained=True, num_classes=2, drop_rate=args.drop_rate)
    elif args.arch == 'dense161': 
        model = DenseUnet(arch='161', pretrained=True, num_classes=2, drop_rate=args.drop_rate)
    elif args.arch == 'dense201': 
        model = DenseUnet(arch='201', pretrained=True, num_classes=2, drop_rate=args.drop_rate)
    elif args.arch == 'resunet': 
        model = ResUNet(in_ch=3, num_classes=2, relu=False)
    elif args.arch == 'unet': 
        model = UNet(n_channels=3, n_classes=2)
    else:
        raise(RuntimeError('error in building network!'))
    if args.ngpu > 1:
        model = nn.parallel.DataParallel(model, list(range(args.ngpu)))

    n_params = sum([p.data.nelement() for p in model.parameters()])
    logging.info('--- total parameters = {} ---'.format(n_params))

    model = model.cuda()
    #x = torch.zeros((1, 3, 256, 256)).cuda()
    #writer.add_graph(model, x)

    
    ################
    # prepare data #
    ################
    logging.info('--- loading dataset ---')

    train_transform = transforms.Compose([
        ElasticTransform(mode='train'), 
        ToTensor(), 
        Normalize(0.5, 0.5)
        ])
    val_transform = transforms.Compose([
        ElasticTransform('val'), 
        ToTensor(), 
        Normalize(0.5, 0.5)
        ])
    train_image_path = args.train_image_path
    train_target_path = args.train_target_path
    train_set = ABUS_Dataset_2d(
            image_path=train_image_path, 
            target_path=train_target_path, 
            transform=train_transform, 
            mode='train')
    val_set = ABUS_Dataset_2d(
            image_path=args.val_image_path, 
            target_path=args.val_target_path, 
            transform=val_transform, 
            mode='val')
    batch_size = args.ngpu*args.batchsize
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    kwargs = {'num_workers': 0, 'pin_memory': True, 'worker_init_fn':worker_init_fn}
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)


    #####################
    # optimizer & loss  #
    #####################
    logging.info('--- configing optimizer & losses ---')
    lr = args.lr
    lr_scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.n_epochs, len(train_loader))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    loss_fn = {}
    loss_fn['surface_loss'] = SurfaceLoss()
    loss_fn['ti_loss'] = TILoss()
    loss_fn['dice_loss'] = DiceLoss()
    loss_fn['mse_loss'] = nn.MSELoss()
    loss_fn['mask_mse_loss'] = MaskMSELoss()


    #####################
    #   strat training  #
    #####################
    logging.info('--- start training ---')

    best_pre = 0.
    err_best = 0.
    nTrain = len(train_set)

    for epoch in range(args.start_epoch, args.n_epochs + 1):
        train(args, epoch, model, train_loader, optimizer, loss_fn, writer, lr_scheduler)
        if epoch == 1 or epoch % 5 == 0:
            dice = val(args, epoch, model, val_loader, optimizer, loss_fn, writer)
            is_best = False
            if dice > best_pre:
                is_best = True
                best_pre = dice

            if is_best or epoch % 10 == 0:
                save_checkpoint({'epoch': epoch,
                                 'state_dict': model.state_dict(),
                                 'best_pre': best_pre},
                                  is_best, 
                                  args.save, 
                                  args.arch)
    writer.close()


def train(args, epoch, model, train_loader, optimizer, loss_fn, writer, lr_scheduler):
    model.train()
    nProcessed = 0
    batch_size = args.ngpu * args.batchsize
    nTrain = len(train_loader.dataset)
    loss_list = []
    dice_loss_list = []

    for batch_idx, sample in enumerate(train_loader):
        # read data
        data, target = sample['image'], sample['target']
        data, target = Variable(data.cuda()), Variable(target.cuda(), requires_grad=False)
        
        # feed to model 
        out = model(data)
        out = F.softmax(out, dim=1)
        dice_loss = loss_fn['dice_loss'](target, out)
        loss = dice_loss 
        dice_loss_list.append(dice_loss.item())
        loss_list.append(loss.item())

        # back propagation
        lr = lr_scheduler(optimizer, batch_idx, epoch, 0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # show some result on tensorboard 
        writer.add_scalar('lr', lr, epoch)
        nProcessed += len(data)
        partialEpoch = epoch + batch_idx / len(train_loader)
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(train_loader),
            loss.item()))

        # show images on tensorboard
        with torch.no_grad():
            padding = 10
            nrow = 5
            if batch_idx % 10 == 0:
                # 1. show gt and prediction
                data = (data * 0.5 + 0.5)
                #print('data.max()', data.max())
                #print('data.min()', data.min())
                img = make_grid(data, nrow=nrow, padding=padding).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) * 255
                img = img.astype(np.uint8)
                gt = make_grid(target, nrow=nrow, padding=padding).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                gt_img = label2rgb(gt, img, bg_label=0)
                pre = torch.max(out, dim=1, keepdim=True)[1]
                pre = pre.float()
                pre = make_grid(pre, nrow=nrow, padding=padding).cpu().numpy().transpose(1, 2, 0)[:, :, 0]
                pre_img = label2rgb(pre, img, bg_label=0)
                fig = plt.figure()
                ax = fig.add_subplot(211)
                ax.imshow(gt_img)
                ax.set_title('train ground truth')
                ax = fig.add_subplot(212)
                ax.imshow(pre_img)
                ax.set_title('train prediction')
                fig.tight_layout() 
                writer.add_figure('train_result', fig, epoch)
                fig.clear()

    writer.add_scalar('dice_loss/epoch', float(np.mean(loss_list)), epoch)
    writer.add_scalar('total_loss/epoch',float(np.mean(loss_list)), epoch)

        
def val(args, epoch, model, val_loader, optimizer, loss_fn, writer):
    model.eval()
    mean_dice = []
    mean_loss = []
    mean_precision = []
    mean_recall = []
    with torch.no_grad():
        for sample in tqdm.tqdm(val_loader):
            data, target = sample['image'], sample['target']
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)

            out = model(data)
            out = F.softmax(out, dim=1)
            loss = loss_fn['dice_loss'](target, out)
            out_new = out.max(1)[1]
            dice = DiceLoss.dice_coeficient(out_new, target)
            precision, recall = confusion(out_new, target)

            mean_precision.append(precision.item())
            mean_recall.append(recall.item())
            mean_dice.append(dice.item())
            mean_loss.append(loss.item())

        # show the last sample
        # 1. show gt and prediction
        data = (data * 0.5 + 0.5)
        img = make_grid(data, padding=20).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) * 255
        img = img.astype(np.uint8)
        gt = make_grid(target, padding=20).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
        pre = torch.max(out, dim=1, keepdim=True)[1]
        pre = pre.float()
        pre = make_grid(pre, padding=20).cpu().numpy().transpose(1, 2, 0)[:, :, 0]
        pre_img = label2rgb(pre, img, bg_label=0)
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.imshow(img)
        contours = measure.find_contours(gt, 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], 'g')
        ax.set_title('val_ground_truth')
        ax = fig.add_subplot(212)
        ax.imshow(pre_img)
        ax.set_title('val_prediction')
        fig.tight_layout() 
        writer.add_figure('val_result', fig, epoch)
        fig.clear()

        writer.add_scalar('val_dice/epoch', float(np.mean(mean_dice)), epoch)
        writer.add_scalar('val_loss/epoch', float(np.mean(mean_loss)), epoch)
        writer.add_scalar('val_precisin/epoch', float(np.mean(mean_precision)), epoch)
        writer.add_scalar('val_recall/epoch', float(np.mean(mean_recall)), epoch)

        return np.mean(mean_dice)


if __name__ == '__main__':
    main()
