import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3' 
import tqdm
import argparse 
import time 
import shutil
import torch 
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from skimage.color import label2rgb 
from skimage import measure
import matplotlib.pyplot as plt
import setproctitle
import numpy as np 
from models.denseunet import DenseUnet
from models.resunet import ResUNet
from models.unet import UNet
from utils.logger import Logger
from utils.loss import DiceLoss, SurfaceLoss, TILoss, MaskDiceLoss, MaskMSELoss
#from utils.ramps import sigmoid_rampup #https://github.com/yulequan/UA-MT/blob/master/code/utils/ramps.py
from dataset.uatm_dataset_2d import ABUS_Dataset_2d, ElasticTransform, ToTensor, Normalize, CenterCrop, RandomCrop
#from dataset.isic_dataset_2d import ABUS_Dataset_2d, ElasticTransform, ToTensor, Normalize
plt.switch_backend('agg')

def get_args():
    print('initing args------')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=12)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--gpu_idx', default=1, type=str)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W')
    parser.add_argument('--save')
    parser.add_argument('--opt', type=str, default='adam', choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--sample_k', '-k', default=None, type=int) #'number of sampled images'
    parser.add_argument('--max_val', default=1, type=float) # maxmum of ramp-up function 
    parser.add_argument('--train_method', default='super', choices=('super', 'semisuper'))
    parser.add_argument('--arch', default='dense121', type=str) #architecture
    parser.add_argument('--drop_rate', default=0.3, type=float) # dropout rate 
    parser.add_argument('--lr', default=1e-4, type=float) # learning rete
    parser.add_argument('--augment', default=True, action='store_true') 
    parser.add_argument('--train_image_path', default='./data/train_data_2d/', type=str)
    parser.add_argument('--train_target_path', default='./data/train_label_2d/', type=str)
    parser.add_argument('--test_image_path', default='./data/test_data_2d/', type=str)
    parser.add_argument('--test_target_path', default='./data/test_label_2d/', type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
    #torch.cuda.set_device(args.gpu_idx)
    return args

def main():
    #############
    # init args #
    #############
    args = get_args()
    train_image_path = args.train_image_path 
    train_target_path = args.train_target_path 
    test_image_path = args.test_image_path
    test_target_path = args.test_target_path

    batch_size = args.ngpu*args.batchsize
    args.cuda = torch.cuda.is_available()
    args.save = args.save or 'work/vnet.base.{}'.format(datestr())
    setproctitle.setproctitle(args.save)

    torch.manual_seed(1)
    if args.cuda:
        torch.cuda.manual_seed(1)

    # writer for tensorboard  
    if args.save:
        idx = args.save.rfind('/')
        log_dir = 'runs' + args.save[idx:]
        print('log_dir', log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = SummaryWriter()
    
    #####################
    # building  network #
    #####################
    print("building network-----")
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
    print('Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    best_prec1 = 0.
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint(epoch {})"
                  .format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    model = model.cuda()

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    # define a logger and write information 
    logger = Logger(os.path.join(args.save, 'log.txt')) 
    logger.print3('batch size is %d' % args.batchsize)
    logger.print3('nums of gpu is %d' % args.ngpu)
    logger.print3('num of epochs is %d' % args.n_epochs)
    logger.print3('start-epoch is %d' % args.start_epoch)
    logger.print3('weight-decay is %e' % args.weight_decay)
    logger.print3('optimizer is %s' % args.opt)
    
    ################
    # prepare data #
    ################
    #train_transform = transforms.Compose([RandomCrop((128, 512)), ElasticTransform(mode='train'), ToTensor(), Normalize(0.5, 0.5)])
    #test_transform = transforms.Compose([CenterCrop((128, 512)), ElasticTransform('test'), ToTensor(), Normalize(0.5, 0.5)])
    train_transform = transforms.Compose([ElasticTransform(mode='train'), ToTensor(), Normalize(0.5, 0.5)])
    test_transform = transforms.Compose([ElasticTransform('test'), ToTensor(), Normalize(0.5, 0.5)])

    # tarin dataset
    print("loading train set --- ")
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    train_set = ABUS_Dataset_2d(image_path=train_image_path, target_path=train_target_path, transform=train_transform, sample_k=args.sample_k, seed=1)
    test_set = ABUS_Dataset_2d(image_path=test_image_path, target_path=test_target_path, transform=test_transform, mode='test')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)

    #############
    # optimizer #
    #############
    lr = args.lr
    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), weight_decay=args.weight_decay)

    # loss function
    loss_fn = {}
    loss_fn['surface_loss'] = SurfaceLoss()
    loss_fn['ti_loss'] = TILoss()
    loss_fn['dice_loss'] = DiceLoss()
    loss_fn['mse_loss'] = nn.MSELoss()
    loss_fn['mask_mse_loss'] = MaskMSELoss()

    ############
    # training #
    ############
    err_best = 0.
    nTrain = len(train_set)
    for epoch in range(args.start_epoch, args.n_epochs + 1):
        if args.opt == 'sgd':
            if (epoch+1) % 30 == 0:
                lr *= 0.1
        if args.opt == 'adam':
            if (epoch+1) % 30 == 0:
                if (epoch+1) % 60 == 0:
                    lr *= 0.2
                else:
                    lr *= 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


        writer.add_scalar('lr/epoch', lr, epoch)
        train(args, epoch, model, train_loader, optimizer, loss_fn, writer)
        dice = test(args, epoch, model, test_loader, optimizer, loss_fn, logger, writer)


        is_best = False
        if dice > best_prec1:
            is_best = True
            best_prec1 = dice

        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_prec1},
                        is_best, args.save, "unet")
    writer.close()



def train(args, epoch, model, train_loader, optimizer, loss_fn, writer):
    batch_size = args.ngpu * args.batchsize
    model.train()

    nProcessed = 0
    nTrain = len(train_loader.dataset)
    loss_list = []
    dice_loss_list = []
    surface_loss_list = []
    for batch_idx, sample in enumerate(train_loader):
        # read data
        data, target, bounds = sample['image'], sample['target'], sample['bounds']
        data, target, bounds = Variable(data.cuda()), Variable(target.cuda(), requires_grad=False), Variable(bounds.cuda(), requires_grad=False)
        #print('data.shape', data.shape)
        #print('target.shape:', target.shape)
        
        # feed to model 
        out = model(data)
        out = F.softmax(out, dim=1)
        dice_loss = loss_fn['dice_loss'](target, out)
        #surface_loss = loss_fn['surface_loss'](out, target, bounds)
        #w = linear_rampup(epoch, 100)
        #w = Variable(torch.FloatTensor([w]).cuda(), requires_grad=False)
        #loss = w * surface_loss + (1-w) * dice_loss 
        loss = dice_loss 
        dice_loss_list.append(dice_loss.item())
        loss_list.append(loss.item())
        #surface_loss_list.append(surface_loss.item())

        #dice = DiceLoss.dice_coeficient(out.max(1)[1], target) 
        #precision, recall = confusion(out.max(1)[1], target) #target = target.view((out.shape[0], target.numel()//out.shape[0]))

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # show some result on tensorboard 
        nProcessed += len(data)
        partialEpoch = epoch + batch_idx / len(train_loader)
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(train_loader),
            loss.item()))

        # show images on tensorboard
        with torch.no_grad():
            index = torch.ones(1).long().cuda()
            index0 = torch.zeros(1).long().cuda()
            if batch_idx % 10 == 0:
                # 1. show gt and prediction
                data = (data * 0.5 + 0.5)
                img = make_grid(data, padding=20).cpu().detach().numpy().transpose(1, 2, 0)
                gt = make_grid(target, padding=20).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                gt_img = label2rgb(gt, img, bg_label=0)

                pre = torch.max(out, dim=1, keepdim=True)[1]
                pre = pre.float()
                pre = make_grid(pre, padding=20).cpu().numpy().transpose(1, 2, 0)[:, :, 0]
                pre_img = label2rgb(pre, img, bg_label=0)

                fig = plt.figure()
                ax = fig.add_subplot(211)
                ax.imshow(gt_img)
                ax.set_title('train ground truth')

                ax = fig.add_subplot(212)
                ax.imshow(pre_img)
                ax.set_title('train prediction')
                fig.tight_layout() 
                writer.add_figure('train result', fig, epoch)
                fig.clear()

                bounds_ = make_grid(bounds, padding=20).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                #print('bounds.shape', bounds.shape)
                #print('bounds_.shape', bounds_.shape)
                #print('target.shape', target.shape)
                fig = plt.figure()
                ax = fig.add_subplot(211)
                ax.imshow(gt)
                ax.set_title('train gt')

                ax = fig.add_subplot(212)
                ax.imshow(bounds_)
                ax.set_title('train bounds')

                writer.add_figure('bounds', fig, epoch)
                plt.tight_layout()
                fig.clear()

    writer.add_scalar('dice_loss/epoch', float(np.mean(loss_list)), epoch)
    writer.add_scalar('total_loss/epoch',float(np.mean(loss_list)), epoch)
    #writer.add_scalar('w/epoch', w, epoch)

        
def test(args, epoch, model, test_loader, optimizer, loss_fn, logger, writer):
    model.eval()
    mean_dice = []
    mean_loss = []
    mean_precision = []
    mean_recall = []
    with torch.no_grad():
        for sample in tqdm.tqdm(test_loader):
            data, target, bounds = sample['image'], sample['target'], sample['bounds']
            if args.cuda:
                data, target, bounds = data.cuda(), target.cuda(), bounds.cuda
            data, target, bounds = Variable(data, requires_grad=False), Variable(target, requires_grad=False), Variable(target, requires_grad=False)

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
        img = make_grid(data, padding=20).cpu().detach().numpy().transpose(1, 2, 0)
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
        ax.set_title('test ground truth')
        ax = fig.add_subplot(212)
        ax.imshow(pre_img)
        ax.set_title('test prediction')
        fig.tight_layout() 
        writer.add_figure('test result', fig, epoch)
        fig.clear()

        writer.add_scalar('test_dice/epoch', float(np.mean(mean_dice)), epoch)
        writer.add_scalar('test_loss/epoch', float(np.mean(mean_loss)), epoch)
        writer.add_scalar('test_precisin/epoch', float(np.mean(mean_precision)), epoch)
        writer.add_scalar('test_recall/epoch', float(np.mean(mean_recall)), epoch)
        return np.mean(mean_dice)

def confusion(y_pred, y_true):
    '''
    get precision and recall
    '''
    y_pred = y_pred.float().view(-1) 
    y_true = y_true.float().view(-1)
    #print('y_pred.shape', y_pred.shape)
    #print('y_true.shape', y_true.shape)
    smooth = 1. 
    #y_pred_pos = np.clip(y_pred, 0, 1)
    y_pred_pos = y_pred
    y_pred_neg = 1 - y_pred_pos
    #y_pos = np.clip(y_true, 0, 1)
    y_pos = y_true
    y_neg = 1 - y_true

    tp = torch.dot(y_pos, y_pred_pos)
    fp = torch.dot(y_neg, y_pred_pos)
    fn = torch.dot(y_pos, y_pred_neg)

    prec = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return prec, recall

def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def linear_rampup(current, rampup_length):
    """Linear rampup"""
    w = 0.
    if current <= 2:
        return w 
    else:
        w = current * 0.01

    if w >= 1:
        w = 0.99

    return w

def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')

if __name__ == '__main__':
    main()
