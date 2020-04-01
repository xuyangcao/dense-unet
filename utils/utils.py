import os
import cv2
import time
import torch
import shutil
import numpy as np
from medpy import metric
from skimage.transform import resize

def confusion(y_pred, y_true):
    '''
    get precision and recall
    '''
    y_pred = y_pred.float().view(-1) 
    y_true = y_true.float().view(-1)

    smooth = 1. 

    y_pred_pos = y_pred
    y_pred_neg = 1 - y_pred_pos
    y_pos = y_true
    y_neg = 1 - y_true

    tp = torch.dot(y_pos, y_pred_pos)
    fp = torch.dot(y_neg, y_pred_pos)
    fn = torch.dot(y_pos, y_pred_neg)

    prec = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return prec, recall

def draw_results(img, label, pred):
    _, contours, _ = cv2.findContours(label, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    pred = resize(pred, label.shape).astype(label.dtype)
    _, contours, _ = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (255, 0, 0), 2)

    return img

def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    filename = 'checkpoint_' + str(state['epoch']) + '.pth.tar'
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')

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

def get_metrics(pred, gt, voxelspacing=(0.21, 0.21)):
    r""" 
    Get statistic metrics of segmentation

    These metrics include: Dice, Jaccard, Hausdorff Distance, 95% Hausdorff Distance, 
    , Pixel Wise Accuracy, Precision and Recall.

    If the prediction result is 0s, we set hd, hd95, 10.0 to avoid errors.

    Parameters:
    -----------

        pred: 2D numpy ndarray
            binary prediction results 

        gt: 2D numpy ndarray
            binary ground truth

        voxelspacing: tuple of 2 floats. default: (0.21, 0.21)
            voxel space of 2D image

    Returns:
    --------

        metrics: dict of 7 metrics 
            dict{dsc, jc, hd, hd95, precision, recall, acc}

    """

    dsc = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    precision = metric.binary.precision(pred, gt)
    recall = metric.binary.recall(pred, gt)

    acc = (pred == gt).sum() / len(gt.flatten())

    if np.sum(pred) == 0:
        #print('=> prediction is 0s! ')
        hd = 10 
        hd95 = 10
    else:
        hd = metric.binary.hd(pred, gt, voxelspacing=voxelspacing)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxelspacing)

    metrics = {'dsc': dsc, 'jc': jc, 'hd': hd, 'hd95': hd95, 
                'precision':precision, 'recall':recall, 'acc':acc} 
    return metrics 
