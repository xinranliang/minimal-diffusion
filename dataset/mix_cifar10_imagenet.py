import os
import pickle
import tempfile
import glob
from shutil import copyfile

import torchvision
from tqdm.auto import tqdm

import numpy as np
import random
from PIL import Image
import cv2
import scipy, scipy.io
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from utils import rgb_to_gray

CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

def check_torchdataset_order():
    round1 = datasets.ImageFolder(
                root = "/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/train",
                transform=transforms.Compose([transforms.ToTensor(),]),
                target_transform=None
            )
    round2 = datasets.ImageFolder(
                root = "/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/train",
                transform=transforms.Compose([transforms.ToTensor(),]),
                target_transform=None
            )
    assert len(round1) == len(round2)
    for index in range(len(round1))[:100]:
        img1, target1 = round1.__getitem__(index)
        img2, target2 = round2.__getitem__(index)
        assert torch.equal(img1, img2)
        assert target1 == target2
    print("All asserted equal in same order")

def split_fixcolor(date):
    transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    fixcolor = 15000
    gray_num = [0, 30000, 60000, 90000, 120000, 150000, 180000, 210000, 240000]

    os.makedirs(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/index_split", date), exist_ok=True)

    for number in gray_num:
        mydata = Mix_CIFAR10ImageNet(
            root="/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/train",
            transform=transform_train,
            target_transform=None,
            fix="color",
            color_num=fixcolor,
            gray_num=number,
            date=date,
            split=True
        )


class Mix_CIFAR10ImageNet(datasets.ImageFolder):
    def __init__(
        self,
        root, # dataset root folder
        transform, # data augmentation
        target_transform, # default None
        fix, # ["color", "gray"] default fix color and add gray
        color_num, # base dataset size
        gray_num, # other dataset size
        date, # date of experiments to handle multiple runs
        split=False # whether to split index or directly load
    ):
        super().__init__(root=root, transform=transform, target_transform=target_transform)

        # load color-gray split index
        self.color_num = color_num
        self.gray_num = gray_num

        if split:
            if self.gray_num > 0:
                # stick to original already splitted index
                with open(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/index_split", date, "color{}_gray0_index.pkl".format(color_num)), "rb") as f:
                    file_load = pickle.load(f)
                self.color_index = file_load["color_index"]
                # random sample w/o replacement from remaining indices
                self.gray_index = [no_idx for no_idx in range(len(self.samples)) if no_idx not in self.color_index]
                self.gray_index = np.array(self.gray_index, dtype=int)
                assert self.gray_num <= len(self.gray_index)
                self.gray_index = np.random.choice(self.gray_index, size=self.gray_num, replace=False)

            elif self.gray_num == 0:
                self.color_index = np.random.choice(len(self.samples), size=self.color_num, replace=False)
                self.gray_index = []
            
            else:
                assert self.gray_num >= 0, "Number of samples has to be specified as non-negative value"
            
            idx_dict = {"color_index": self.color_index, "gray_index": self.gray_index}

            file_path = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/index_split", date, "color{}_gray{}_index.pkl".format(color_num, gray_num))
            with open(file_path, "wb") as f:
                pickle.dump(idx_dict, f)

        else:
            file_path = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/index_split", date, "color{}_gray{}_index.pkl".format(color_num, gray_num))
            with open(file_path, "rb") as f:
                file_load = pickle.load(f)
            print("Loading color/gray index from file path {}".format(file_path))
            self.color_index = file_load["color_index"]
            self.gray_index = file_load["gray_index"]

            assert len(self.color_index) == self.color_num
            assert len(self.gray_index) == self.gray_num

            # random shuffle all color and gray index
            # both are numpy array (color_number, ) (gray_number, )
            self.image_index = np.concatenate((self.color_index, self.gray_index), axis=0).astype(int)
            random.shuffle(self.image_index)
            assert len(self.image_index) == self.color_num + self.gray_num
        
    
    def __len__(self):
        return self.color_num + self.gray_num

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        img_idx = self.image_index[index]
        img, target = super().__getitem__(img_idx) # in tensor

         # decide color or grayscale
        if img_idx in self.gray_index:
            assert img_idx not in self.color_index
            # get image
            path, _ = self.samples[img_idx]
            sample = self.loader(path)
            # color -> gray
            img = sample.convert("L")
            # transform
            if self.transform is not None:
                img = self.transform(img)
            img = img.expand(3, -1, -1)
            # assert torch.equal(img[0], img[1])
            # assert torch.equal(img[1], img[2])
            return img, target
        else:
            assert img_idx in self.color_index
            return img, target



if __name__ == "__main__":
    split_fixcolor(date="2023-04-03")