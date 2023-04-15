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
    gray_num = [0, 7500, 15000, 22500, 30000, 60000, 90000, 120000, 150000, 180000, 210000, 240000]

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

def split_fixgray(date):
    transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    fixgray = 15000
    color_num = [0, 7500, 15000, 22500, 30000, 60000, 90000, 120000, 150000, 180000, 210000, 240000]

    os.makedirs(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/index_split", date), exist_ok=True)

    for number in color_num:
        mydata = Mix_CIFAR10ImageNet(
            root="/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/train",
            transform=transform_train,
            target_transform=None,
            fix="gray",
            color_num=number,
            gray_num=fixgray,
            date=date,
            split=True
        )

def split_random_baseline(date):
    transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    num_train = [25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000, 225000, 250000]
    os.makedirs(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/index_split", date), exist_ok=True)

    for number in num_train:
        mydata = Mix_CIFAR10ImageNet(
            root="/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/train",
            transform=transform_train,
            target_transform=None,
            fix="none",
            color_num=number,
            gray_num=0,
            date=date,
            split=True
        )

def split_domain_classifier(train_split, test_split):
    full_length = 260000 # total 270000 and exclude 10000 cifar10-test
    indices = list(range(full_length))
    split = int(np.floor(test_split * full_length))
    # random shuffle indices
    np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]
    index_dict = {
        "train_index": train_idx,
        "test_index": test_idx
    }
    folder_path = "/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/index_split/domain_classifier"
    os.makedirs(folder_path, exist_ok=True)

    with open(os.path.join(folder_path, "train{}_test{}_index.pkl".format(train_split, test_split)), "wb") as f:
        pickle.dump(index_dict, f)
    
    return 


class Mix_CIFAR10ImageNet(datasets.ImageFolder):
    def __init__(
        self,
        root, # dataset root folder
        transform, # data augmentation
        target_transform, # default None
        fix, # ["color", "gray", "none"] default fix color and add gray
        color_num, # base dataset size
        gray_num, # other dataset size
        date, # date of experiments to handle multiple runs
        split=False # whether to split index or directly load
    ):
        super().__init__(root=root, transform=transform, target_transform=target_transform)
        self.fix = fix

        # load color-gray split index
        self.color_num = color_num
        self.gray_num = gray_num

        if split and (fix == "color" or fix == "gray"):
            if fix == "color":
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
            
            elif fix == "gray":
                if self.color_num > 0:
                    # stick to original already splitted index
                    with open(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/index_split", date, "gray{}_color0_index.pkl".format(gray_num)), "rb") as f:
                        file_load = pickle.load(f)
                    self.gray_index = file_load["gray_index"]
                    # random sample w/o replacement from remaining indices
                    self.color_index = [no_idx for no_idx in range(len(self.samples)) if no_idx not in self.gray_index]
                    self.color_index = np.array(self.color_index, dtype=int)
                    assert self.color_num <= len(self.color_index)
                    self.color_index = np.random.choice(self.color_index, size=self.color_num, replace=False)

                elif self.color_num == 0:
                    self.gray_index = np.random.choice(len(self.samples), size=self.gray_num, replace=False)
                    self.color_index = []
                
                else:
                    assert self.color_num >= 0, "Number of samples has to be specified as non-negative value"
                
                idx_dict = {"color_index": self.color_index, "gray_index": self.gray_index}

                file_path = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/index_split", date, "gray{}_color{}_index.pkl".format(gray_num, color_num))
                with open(file_path, "wb") as f:
                    pickle.dump(idx_dict, f)
            
            else: 
                raise NotImplementedError

        elif split and fix == "none":
            # handle random baseline case
            self.train_index = np.random.choice(len(self.samples), size=self.color_num, replace=False)
            idx_dict = {"train_index": self.train_index}

            file_path = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/index_split", date, "baseline_train{}_index.pkl".format(color_num))
            with open(file_path, "wb") as f:
                pickle.dump(idx_dict, f)
        
        else:
            if fix == "color" or fix == "gray":
                if fix == "color":
                    file_path = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/index_split", date, "color{}_gray{}_index.pkl".format(color_num, gray_num))
                elif fix == "gray":
                    file_path = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/index_split", date, "gray{}_color{}_index.pkl".format(gray_num, color_num))

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
            
            elif fix == "none":
                file_path = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/index_split", date, "baseline_train{}_index.pkl".format(color_num))
                with open(file_path, "rb") as f:
                    file_load = pickle.load(f)
                print("Loading training samples index from file path {}".format(file_path))
                self.image_index = file_load["train_index"]

                assert len(self.image_index) == self.color_num
                random.shuffle(self.image_index)
        
    
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

        if self.fix == "color" or self.fix == "gray":
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
        
        elif self.fix == "none":
            return img, target
        
        else:
            raise NotImplementedError



if __name__ == "__main__":
    # split_fixcolor(date="2023-04-02")
    # split_fixcolor(date="2023-04-03")
    # split_fixgray(date="2023-04-02")
    # split_fixgray(date="2023-04-03")
    # split_random_baseline(date="2023-04-06")
    # split_random_baseline(date="2023-04-07")
    split_domain_classifier(train_split=0.8, test_split=0.2)
    split_domain_classifier(train_split=0.9, test_split=0.1)