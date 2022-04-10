import os 
import tqdm
import shutil 
import argparse
import setproctitle
import pandas as pd
import numpy as np 
from skimage import measure
from skimage.io import imsave
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.denseunet import DenseUnet
from models.resunet import ResUNet
from models.unet import UNet
from dataset.abus_dataset_2d import ABUS_Dataset_2d, ElasticTransform, ToTensor, Normalize
from utils.utils import get_metrics, draw_results


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_image_path', default='../data/selected_data/', type=str)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--arch', default='dense161', type=str, choices=('dense161', 'dense121', 'dense201', 'unet', 'resunet')) 

    '''frequently changed args'''
    parser.add_argument('--save', default=None, type=str) 
    parser.add_argument('--resume', type=str)
    parser.add_argument('--save_image', action='store_true')

    args = parser.parse_args()
    return args

            
def main():

    # --- init args ---
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '3' 

    # --- building network ---
    if args.arch == 'dense121': 
        model = DenseUnet(arch='121', pretrained=True, num_classes=2)
    elif args.arch == 'dense161': 
        model = DenseUnet(arch='161', pretrained=True, num_classes=2)
    elif args.arch == 'dense201': 
        model = DenseUnet(arch='201', pretrained=True, num_classes=2)
    elif args.arch == 'resunet': 
        model = ResUNet(in_ch=3, num_classes=2, relu=False)
    elif args.arch == 'unet': 
        model = UNet(n_channels=3, n_classes=2)
    else:
        raise(RuntimeError('error in building network!'))

    model = model.cuda()
    model = nn.parallel.DataParallel(model, list(range(args.ngpu)))


    # --- resume trained weights ---
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_pre = checkpoint['best_pre']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

            # --- saving path ---
            if 'best' in args.resume:
                file_name = 'model_best_' + str(checkpoint['epoch'])
            elif 'check' in args.resume:
                file_name = 'checkpoint_{}_result'.format(checkpoint['epoch'])

            if args.save is not None:
                save_path = os.path.join(args.save, file_name) 
                csv_path = os.save
            else:
                save_path = os.path.join(os.path.dirname(args.resume), file_name)
                csv_path = os.path.dirname(args.resume)
            setproctitle.setproctitle(save_path)

            if args.save_image:
                # image path
                args.save_image_path = save_path + '/image' 
                if os.path.exists(args.save_image_path):
                    shutil.rmtree(args.save_image_path)
                os.makedirs(args.save_image_path, exist_ok=True)
                # label path
                args.save_pred_path = save_path + '/label' 
                if os.path.exists(args.save_pred_path):
                    shutil.rmtree(args.save_pred_path)
                os.makedirs(args.save_pred_path, exist_ok=True)
                print('=> saving images in :', save_path)
            else:
                print('we don\'t save any images!')
            # csv path
            csv_file_name = file_name + '.xlsx'
            args.csv_file_name = os.path.join(csv_path, csv_file_name) 
            print('=> saving csv in :', args.csv_file_name)

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        raise(RuntimeError('resume is None!'))


    # --- preparing dataset
    test_transform = transforms.Compose([
                            ElasticTransform(mode='test'),
                            ToTensor(mode='test'), 
                            Normalize(0.5, 0.5, mode='test')
                            ])
    test_set = ABUS_Dataset_2d(
            image_path=args.test_image_path, 
            target_path=args.test_target_path, 
            transform=test_transform, 
            mode='test')
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    kwargs = {'num_workers': 0, 'pin_memory': True, 'worker_init_fn': worker_init_fn}
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)


    # --- testing ---
    test(args, test_loader, model)

def test(args, loader, model):
    model.eval()
    
    dsc_list = []
    jc_list = []
    hd_list = []
    hd95_list = []
    precision_list = []
    recall_list = []
    acc_list = []
    filename_list = []
    area_list = []

    with torch.no_grad():
        for sample in tqdm.tqdm(loader):
            image, label, file_name = sample['image'], sample['target'], sample['file_name']

            image = image.cuda()
            pred = model(image)
            pred = F.softmax(pred, dim=1) 
            pred = pred.max(1)[1]

            image = image[0][0].cpu().numpy()
            image = image * 0.5 + 0.5

            label = label[0][0].cpu().numpy()
            label = label.astype(np.float)
            pred = pred[0].cpu().numpy()
            perd = pred.astype(np.float)

            # get metrics
            metrics = get_metrics(pred, label, voxelspacing=(0.29, 0.14)) 
            dsc_list.append(metrics['dsc'])
            jc_list.append(metrics['jc'])
            hd95_list.append(metrics['hd95'])
            acc_list.append(metrics['acc'])
            hd_list.append(metrics['hd'])
            precision_list.append(metrics['precision'])
            recall_list.append(metrics['recall'])
            filename_list.append(file_name[0]) # ['filename']
            area_list.append(np.sum(label))

            if args.save_image:
                # save images
                image *= 255
                image = image.astype(np.uint8)
                label *= 255
                label = label.astype(np.uint8)
                pred *= 255
                pred = pred.astype(np.uint8)

                #image = draw_results(image, label, pred)
                #imsave(os.path.join(args.save_image_path, file_name[0]), image)
                imsave(os.path.join(args.save_image_path, file_name[0]), image)
                imsave(os.path.join(args.save_pred_path, file_name[0]), pred)
                #break # debug

            
        # --- save statistic result ---
        df = pd.DataFrame()
        df['filename'] = filename_list
        df['dsc'] = np.array(dsc_list)
        df['jc'] = np.array(jc_list)
        df['hd95'] = np.array(hd95_list)
        df['acc'] = np.array(acc_list)
        df['hd'] = np.array(hd_list)
        print(df.describe())
        df['precision'] = np.array(precision_list)
        df['recall'] = np.array(recall_list)
        df['area'] = np.array(area_list)
        df.to_excel(args.csv_file_name)

if __name__ == '__main__':
    main()
