import os, glob
import torch
import numpy as np
import scipy.io
from skimage.color import rgb2lab
import matplotlib.pyplot as plt
import logging
import cv2

import random
import albumentations as A

from os import listdir
from glob import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

COLOURS = {0: [0, 0, 0], 1: [5, 5, 255], 2: [0, 240, 255], 3: [255, 255, 255], 4: [147, 97, 41]}
CLASSES = {0: 'back', 1: 'tool', 2: 'chip', 3: 'wear', 4: 'bue'}

def convert_label(label):

    onehot = np.zeros((1, 50, label.shape[0], label.shape[1])).astype(np.float32)

    ct = 0
    for t in np.unique(label).tolist():
        if ct >= 50:
            break
        else:
            onehot[:, ct, :, :] = (label == t)
        ct = ct + 1

    return onehot


class tool:
    def __init__(self, root, split="train/", color_transforms=None, geo_transforms=None):
        self.mask_dir = os.path.join(root, "tool/data/groundTruth/", split)
        self.img_dir = os.path.join(root, "tool/data/images/", split)

        self.ids = [file.split('.')[0] for file in sorted(listdir(self.img_dir))
                    if not file.startswith('.')]


        self.color_transforms = color_transforms
        self.geo_transforms = geo_transforms

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.img_dir + idx + '*')
        mask_file = 0
        mask_file = glob(self.mask_dir + idx + '*')
        print (idx)
        print (mask_file[0])
        img = cv2.imread(img_file[0])  # Load image from file
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = cv2.resize(img, (1024, 256), cv2.INTER_NEAREST)

        mask = cv2.imread(mask_file[0])
        mask = cv2.resize(mask, (1024, 256), cv2.INTER_NEAREST)
        m = self.rgb_to_onehot(mask)
        mask = np.transpose(m,[2,0,1])
    
        gt = mask.astype(np.int64)
        gt = torch.from_numpy(gt)
        img = img.astype(np.float32)

   
        #gt = convert_label(gt)
        
       
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        gt = gt.reshape(5, -1).float()

        return img, gt


    def __len__(self):
        return len(self.ids)

    def rgb_to_onehot(self, rgb_image, colormap=COLOURS):
        num_classes = len(colormap)
        shape = rgb_image.shape[:2] + (num_classes,)
        encoded_image = np.zeros(shape, dtype=np.int8)
        for i, cls in enumerate(colormap):
                encoded_image[:, :, i] = np.all(rgb_image.reshape((-1, 3)) == colormap[i], axis=1).reshape(shape[:2])
        return encoded_image




