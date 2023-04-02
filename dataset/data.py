import os
import numpy as np
from PIL import Image
import scipy, scipy.io
from easydict import EasyDict
from collections import OrderedDict
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from dataset.cifar10 import CIFAR10_ColorGray, CIFAR10_FixGroup
from dataset.celeba import CelebA_Custom
from dataset.mix_cifar10_imagenet import Mix_CIFAR10ImageNet


def get_metadata(
    name, date,
    fix=None, color=None, grayscale=None, # this is for CIFAR10 2 domains
    other_name=None, fix_num=None, other_num=None, # this is for combine 2 dataset source as 2 domains
):
    if name == "cifar10":
        if fix == "total":
            # fix total number of training images
            assert float(color) + float(grayscale) == 1.0
            metadata = EasyDict(
                {
                    "image_size": 32,
                    "num_classes": 10,
                    "train_images": 50000,
                    "val_images": 10000,
                    "num_channels": 3,
                    "color_ratio": float(color),
                    "grayscale_ratio": float(grayscale),
                    "fix": "total",
                    "split": False,
                    "date": date
                }
            )
        elif fix == "color":
            # specify number of training images for each subgroup
            metadata = EasyDict(
                {
                    "image_size": 32,
                    "num_classes": 10,
                    "train_images": int(color) + int(grayscale),
                    "val_images": 10000,
                    "num_channels": 3,
                    "color_number": int(color),
                    "gray_number": int(grayscale),
                    "fix": fix,
                    "split": False,
                    "date": date
                }
            )
    elif name == "cifar10-other":
        metadata = EasyDict(
            {
                "image_size": 32,
                "num_classes": 10,
                "train_images": fix_num + other_num,
                "val_images": 10000,
                "num_channels": 3,
                "date": date
            }
        )
    elif name == "mix-cifar10-imagenet":
        metadata = EasyDict(
            {
                "image_size": 32,
                "num_classes": 10,
                "train_images": int(color) + int(grayscale),
                "val_images": 10000,
                "color_number": int(color),
                "gray_number": int(grayscale),
                "num_channels": 3,
                "fix": fix,
                "split": False,
                "date": date
            }
        )
    elif name == "celeba":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_attributes": 40,
                "train_images": 202599,
                "val_images": 0,
                "num_channels": 3,
            }
        )
    elif name == "celeba-hq":
        metadata = EasyDict(
            {
                "image_size": 256,
                "num_attributes": 40,
                "train_images": 30000,
                "val_images": 0,
                "num_channels": 3,
            }
        )
    else:
        raise ValueError(f"{name} dataset nor supported!")
    return metadata


# TODO: Add datasets imagenette/birds/svhn etc etc.
def get_dataset(name, data_dir, metadata):
    """
    Return a dataset with the current name. We only support two datasets with
    their fixed image resolutions. One can easily add additional datasets here.

    Note: To avoid learning the distribution of transformed data, don't use heavy
        data augmentation with diffusion models.
    """
    if name == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        """train_set = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=False,
            transform=transform_train,
        )"""
        if metadata.fix == "total":
            train_set = CIFAR10_ColorGray(
                root=os.path.join(data_dir, "cifar10"),
                train=True,
                download=False,
                transform=transform_train,
                target_transform=None,
                color_ratio=metadata.color_ratio,
                grayscale_ratio=metadata.grayscale_ratio,
                split = False,
                date = metadata.date
            )
        elif metadata.fix == "color" or metadata.fix == "gray":
            train_set = CIFAR10_FixGroup(
                root=os.path.join(data_dir, "cifar10"),
                train=True,
                download=False,
                transform=transform_train,
                target_transform=None,
                fix=metadata.fix,
                color_number=metadata.color_number,
                gray_number=metadata.gray_number,
                split=False,
                date = metadata.date
            )
    elif name == "mix-cifar10-imagenet":
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = Mix_CIFAR10ImageNet(
            root = os.path.join(data_dir, "cifar10-imagenet/train"),
            transform=transform_train,
            target_transform=None,
            fix=metadata.fix,
            color_num=metadata.color_number,
            gray_num=metadata.gray_number,
            date=metadata.date,
            split=False
        )
    elif name == "celeba":
        # celebA has a large number of images, avoiding randomcropping.
        transform_train = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = CelebA_Custom(
            root=os.path.join(data_dir, "celebA"),
            target_type="attr",
            transform=transform_train,
        )
    elif name == "celeba-hq":
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.CelebA(
            root=os.path.join(data_dir, "celebA-HQ"),
            split="train",
            target_type = "attr",
            transform=transform_train,
            download=True
        )
    else:
        raise ValueError(f"{name} dataset nor supported!")
    return train_set
