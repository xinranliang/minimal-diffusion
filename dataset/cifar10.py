import os
import pickle
import tempfile

import torchvision
from tqdm.auto import tqdm

import numpy as np
import random
from PIL import Image
import cv2
import scipy, scipy.io
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from utils import rgb_to_grayscale

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


def generate_index_cifar10():
    transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    for color, gray in zip([0.99, 0.95, 0.9, 0.1, 0.05, 0.01], [0.01, 0.05, 0.1, 0.9, 0.95, 0.99]):
        cifar10 = CIFAR10_Custom(
            root="/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10",
            train=True,
            transform=transform_train,
            target_transform=None,
            download=False,
            color_ratio=color,
            grayscale_ratio=gray,
            split=True # whether we're performing one-off splitting
        )

def check_cifar10_index():
    transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    for color, gray in zip([1.0, 0.99, 0.95, 0.9, 0.1, 0.05, 0.01, 0.0], [0.0, 0.01, 0.05, 0.1, 0.9, 0.95, 0.99, 1.0]):
        cifar10 = CIFAR10_Custom(
            root="/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10",
            train=True,
            transform=transform_train,
            target_transform=None,
            download=False,
            color_ratio=color,
            grayscale_ratio=gray,
            split=False # whether we're performing one-off splitting
        )


class CIFAR10_Custom(datasets.CIFAR10):
    def __init__(
        self,
        root,
        train,
        transform,
        target_transform,
        download,
        color_ratio = None, 
        grayscale_ratio = None,
        split = False
    ):
        super().__init__(root, train, transform, target_transform, download=download)

        self.num_classes = 10 # default 10 classes
        assert color_ratio + grayscale_ratio == 1.0
        self.color_ratio = color_ratio
        self.grayscale_ratio = grayscale_ratio

        if split == True and color_ratio is not None and grayscale_ratio is not None:
            # randomly sample color v.s. grayscale classes

            self.color_index = np.random.choice(len(self.data), int(len(self.data) * self.color_ratio), replace=False)
            self.grayscale_index = [no_idx for no_idx in range(len(self.data)) if no_idx not in self.color_index]
            self.grayscale_index = np.array(self.grayscale_index, dtype=int)
            print(self.color_index.shape, self.grayscale_index.shape)
            assert len(self.color_index) == int(len(self.data) * self.color_ratio)
            assert len(self.grayscale_index) == int(len(self.data) * self.grayscale_ratio)

            idx_dict = {"color_index": self.color_index, "gray_index": self.grayscale_index}

            file_path = os.path.join(root, "color_gray_split", "color{}_gray{}_split.pkl".format(self.color_ratio, self.grayscale_ratio))
            with open(file_path, "wb") as f:
                pickle.dump(idx_dict, f)
        
        elif split == False and color_ratio == 1.0 and grayscale_ratio == 0.0:
            # all color
            self.color_index = range(len(self.data))
            self.grayscale_index = []
        elif split == False and color_ratio == 0.0 and grayscale_ratio == 1.0:
            # all gray
            self.color_index = []
            self.grayscale_index = range(len(self.data))
        
        elif split == False:
            split_file_path = os.path.join(root, "color_gray_split", "color{}_gray{}_split.pkl".format(self.color_ratio, self.grayscale_ratio))
            with open(split_file_path, "rb") as f:
                file_load = pickle.load(f)
            print("Loading color/gray split from file path {}".format(split_file_path))
            self.color_index = file_load["color_index"]
            self.grayscale_index = file_load["gray_index"]
            # print(self.color_index.shape)
            # print(self.grayscale_index.shape)
        
        else:
            raise NotImplementedError
            
    

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # decide color or grayscale
        if index in self.grayscale_index:
            assert index not in self.color_index
            img = rgb_to_grayscale(img)
            img = img.astype(np.uint8)
        else:
            assert index in self.color_index
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    

if __name__ == "__main__":
    generate_index_cifar10()
    check_cifar10_index()
    