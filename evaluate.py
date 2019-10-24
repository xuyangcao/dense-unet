import numpy as np 
from skimage.io import imread, imsave
import os
import argparse 
import tqdm
import pandas as pd
from skimage.transform import resize

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, default='./data/test_label_2d')
    parser.add_argument('--pred_path', type=str, default='./results/denseunet161-no-pretrain/')

    args = parser.parse_args()
    # save csv file to the current folder
    if args.pred_path[-1] == '/':
        args.save = args.pred_path[:-1] + '.csv'
    else:
        args.save = args.pred_path + '.csv'

    return args

def main():
    args = get_args()

    filenames = [filename for filename in os.listdir(args.pred_path) if filename.endswith('.png')]
    ac_list = []
    se_list = []
    sp_list = [] 
    ja_list = []
    di_list = []
    for filename in tqdm.tqdm(filenames):
        gt_img = imread(os.path.join(args.gt_path, filename))
        pre_img = imread(os.path.join(args.pred_path, filename))
        gt_img = resize(gt_img, pre_img.shape)
        #print(pre_img.shape)
        #print(gt_img.shape)
        gt_img[gt_img > 0] = 1.
        pre_img[pre_img > 0] = 1.
        ac, se, sp, ja, di = confusion(pre_img, gt_img)
        ac_list.append(ac)
        se_list.append(se)
        sp_list.append(sp)
        ja_list.append(ja)
        #di_list.append(di)
        di_list.append(dice_coeficient(pre_img, gt_img))

    df = pd.DataFrame()
    df['name'] = filenames
    df['ac'] = np.array(ac_list)
    df['se'] = np.array(se_list) 
    df['sp'] = np.array(sp_list) 
    df['ja'] = np.array(ja_list) 
    df['di'] = np.array(di_list) 
    print(df.mean())
    print(df.std())
    df.to_csv(args.save)

def dice_coeficient(output, target):
    iflat = output.flatten().astype(np.float64) 
    tflat = target.flatten().astype(np.float64) 
    smooth = 1. 

    intersection = np.dot(iflat, tflat)
    dice = (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

    return dice 

def confusion(y_pred, y_true):
    '''
    get precision and recall
    '''
    y_pred = y_pred.flatten().astype(np.float64) 
    y_true = y_true.flatten().astype(np.float64) 
    smooth = 1. 
    #y_pred_pos = np.clip(y_pred, 0, 1)
    y_pred_pos = y_pred
    y_pred_neg = 1 - y_pred_pos
    #y_pos = np.clip(y_true, 0, 1)
    y_pos = y_true
    y_neg = 1 - y_true

    tp = np.dot(y_pos, y_pred_pos)
    fp = np.dot(y_neg, y_pred_pos)
    fn = np.dot(y_pos, y_pred_neg)
    tn = np.dot(y_neg, y_pred_neg)
    #print(tp)

    ac = (tp + tn) / (tp + fp + tn + fn)
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    ja = tp / (tp + fn + fp)
    di = (2 * tp + 1) / (2 * tp + fn + fp + 1)

    return ac, se, sp, ja, di

if __name__ == '__main__':
    main()
