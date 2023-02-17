import random
import cv2
import numpy as np
import glob
import torch

from torch.utils.data import Dataset


class_to_color = {0: [0, 0, 0], 1: [255, 5, 5], 2: [255, 240, 0], 3: [255, 255, 255], 4: [41, 97, 147]}

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


class ImageDataset(Dataset):
    """
    This dataset class can load images
    """
    def __init__(self, id, root, enc, considered_classes, iWindowSize=48,
                 max_samples_per_picture_per_class_train=50):
        self.id = id
        self.gt_dir = os.path.join(root, "BSDS500/data/groundTruth", split)
        self.img_dir = os.path.join(root, "BSDS500/data/images", split)
        
        self.enc = enc
        self.considered_classes = considered_classes
        self.iWindowSize = iWindowSize
        self.max_samples_per_picture_per_class_train = max_samples_per_picture_per_class_train

        image_file = glob.glob(self.imgs_dir + self.id + '*')[0]
        mask_file = glob.glob(self.gt_dir + self.id + '*')[0]

        # Load image and mask
        image = cv2.imread(image_file)  # Load image from file
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB colorspace
        # image = cv2.resize(image, (512, 256), cv2.INTER_CUBIC)
        mask = cv2.imread(mask_file)  # Load mask from file
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # Convert to RGB colorspace
        # mask = cv2.resize(mask, (512, 256), cv2.INTER_CUBIC)

        def conv(mask):
            cl = []
            for img in mask:
                label = np.zeros((img.shape[0], img.shape[1]))

                for img_class in self.considered_classes:
                    label[np.where((img == class_to_color[img_class]).all(axis=2))] = img_class
                cl.append(label)

            return np.asarray(cl)

        classes = conv([mask])

        

        y = self.enc.transform(y.reshape(-1, 1))

        X = np.asarray(X)
        y = np.asarray(y)

        # Shuffle entries
        c = list(zip(X, y))
        random.shuffle(c)
        X, y = zip(*c)

        X = np.asarray(X)
        y = np.asarray(y)

        self.X = X.swapaxes(1, 3).astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]
