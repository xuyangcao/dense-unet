import os
from glob import glob
import numpy as np
from numpy.random import randint
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.color import grey2rgb 
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt as distance

import torch
import torch.utils.data as data 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import Augmentor

__all__ = ['ElasticTransform', 'ToTensor', 'ABUS_Dataset_2d', 'Normalize', 'CenterCrop', 'RandomCrop']

def _target2bounds(target):
   posmask = target.astype(np.bool)
   negmask = ~posmask
   dist_map = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
   return dist_map 

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['target']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1]: 
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)

        (w, h) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]

        return {'image': image, 'target': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, bounds = sample['image'], sample['target'], sample['bounds']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)

        (w, h) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        bounds = bounds[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        return {'image': image, 'target': label, 'bounds':bounds}

class ElasticTransform(object):
    def __init__(self, mode='train'):
        self.mode = mode

    def __call__(self, sample):
        #print(self.mode)
        if self.mode == 'train' or self.mode == 'val' or self.mode == 'test':
            image, target = sample['image'], sample['target']
            images = [[image, target]]

            p = Augmentor.DataPipeline(images)
            p.resize(probability=1, width=512, height=128)
            # random flip  
            #p.flip_left_right(probability=0.5)
            #p.rotate(0.5, 10, 10)
            #p.zoom_random(0.5, 0.8)
            sample_aug = p.sample(1)
            
            sample['image'] = grey2rgb(sample_aug[0][0]).transpose((2, 0, 1))
            sample['target'] = sample_aug[0][1]

            return sample


class ToTensor(object):
    def __init__(self, mode='train'):
        self.mode = mode
    
    def __call__(self, sample):
        if self.mode == 'train' or self.mode == 'val' or self.mode == 'test':
            image, target = sample['image'], sample['target']
            image = image.astype(np.float32)
            image = torch.from_numpy(image)
            target = np.expand_dims(target, 0)
            target = target.astype(np.float32)
            target = torch.from_numpy(target)

            # transverse tensor to 0~1 
            if isinstance(image, torch.ByteTensor): 
                image = image.float().div(255)

            sample['image'] = image
            sample['target'] = target

            return sample 

        else:
            raise(RuntimeError('error in ToTensor'))


class Normalize(object):
    def __init__(self, mean, std, mode='train'):
        self.mode = mode 
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        if self.mode == 'train' or self.mode == 'test' or self.mode == 'val':
            image = sample['image']
            image = (image - self.mean) / self.std

            sample['image'] = image

            return sample

class ABUS_Dataset_2d(data.Dataset):
    ''' ABUS_Dataset class, return 2d transverse images and targets ''' 
    def __init__(self, image_path=None, target_path=None, transform=None, mode='train'): 
        file_names = ABUS_Dataset_2d.get_all_filenames(image_path)
        if len(file_names) == 0:
            raise(RuntimeError("Found 0 images in : " + os.path.join(image_path) + "\n"))

        self.file_names = file_names
        self.mode = mode
        self.image_path = image_path
        self.target_path = target_path
        self.transform = transform

    def __getitem__(self, index):
        # get sample filename
        file_name = self.file_names[index]

        # load image
        image = ABUS_Dataset_2d.load_image(self.image_path, file_name)
        target = ABUS_Dataset_2d.load_image(self.target_path, file_name)
        if target.max() != 1: # transform from 255 to 1
            target[target != 0] = 1.

        # transform 
        sample = {'image':image, 'target':target}
        if self.transform is not None:
            sample = self.transform(sample)

        sample['file_name'] = file_name
        return sample


    def __len__(self):
        return len(self.file_names)

    def get_target_mean(self):
        return self._target_means

    @staticmethod
    def get_all_filenames(image_path):
        all_filenames = [file_name for file_name in os.listdir(image_path) if file_name.endswith('png')]

        return all_filenames

        
    @staticmethod
    def load_image(file_path, file_name):
        full_name = os.path.join(file_path, file_name)
        img = imread(full_name) 

        # we don't normalize image when loading them, because Augmentor will raise error
        # if nornalize, normalize origin image to mean=0,std=1.
        #if is_normalize:
        #    img = img.astype(np.float32)
        #    mean = np.mean(img)
        #    std = np.std(img)
        #    img = (img - mean) / std

        return img 


if __name__ == '__main__':
    # test bjtu_dataset_2d
    image_path = '../abus_data_2d/test_data_2d/'
    target_path = '../abus_data_2d/test_label_2d/'
    
    transform = transforms.Compose([ToTensor(),
                                    Normalize(0.5, 0.5)
                                    ])

    train_set = ABUS_Dataset_2d(image_path, target_path, transform, sample_k=100, seed=1, mode='test')
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
    a = []
    for sample in train_loader:
        image, target = sample['image'], sample['target']
        print('target>0', target[:, 0, 0, 0]>=0)
        a += list((target[:,0,0,0]>=0).numpy())
        print('image shape: ', image.shape)
        print('label shape: ', target.shape)
        #cv2.imshow('', image[0].numpy().transpose(1, 2, 0)) 
        #cv2.imshow('1', image[1].numpy().transpose(1, 2, 0))
        #cv2.waitKey(1000)
    print(a)
    print(len(a))
    print(np.sum(a))
