import os
import numpy as np
from PIL import Image
import scipy, scipy.io
from easydict import EasyDict
from collections import OrderedDict
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from dataset.cifar10 import CIFAR10_ColorGray, CIFAR10_FixGroup, CIFAR_SuperClass
from dataset.celeba import CelebA_AttrCond
from dataset.fairface import FairFace_Gender, FairFace_Base
from dataset.mix_cifar10_imagenet import Mix_CIFAR10ImageNet
from dataset.cifar_imagenet import CIFAR_ImageNet
from dataset.mnist import MNIST_FLIP


def get_metadata(
    name, date,
    fix=None, color=None, grayscale=None, # this is for CIFAR10 2 domains
    num_cifar=None, num_imagenet=None, # this is for combine 2 dataset source as 2 domains
    num_train_baseline=None, # this is for combine 2 dataset source as 1 domain - baseline for above setting
    flip_left=None, flip_right=None, # this is for testing mnist dataset
    semantic_group=None, front_ratio=None, back_ratio=None, # this is for grouping cifar samples into super classes
    female_ratio=None, male_ratio=None, # this is for gender domain in celeba and fairface dataset
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
                    "train_images": color + grayscale,
                    "val_images": 10000,
                    "num_channels": 3,
                    "color_number": color,
                    "gray_number": grayscale,
                    "fix": fix,
                    "split": False,
                    "date": date
                }
            )
    elif name == "cifar-imagenet":
        metadata = EasyDict(
            {
                "image_size": 32,
                "num_classes": 10,
                "train_images": num_cifar + num_imagenet,
                "val_images": 10000,
                "num_cifar": num_cifar,
                "num_imagenet": num_imagenet,
                "num_channels": 3,
                "date": date
            }
        )
    elif name == "cifar-imagenet-check":
        metadata = EasyDict(
            {
                "image_size": 32,
                "num_classes": 20,
                "train_images": num_cifar + num_imagenet,
                "val_images": 10000,
                "num_cifar": num_cifar,
                "num_imagenet": num_imagenet,
                "num_channels": 3,
                "date": date
            }
        )
    elif name == "mix-cifar10-imagenet" and (fix == "total" or fix == "color" or fix == "gray" or fix == "half"):
        metadata = EasyDict(
            {
                "image_size": 32,
                "num_classes": 10,
                "train_images": color + grayscale,
                "val_images": 10000,
                "color_number": color,
                "gray_number": grayscale,
                "num_channels": 3,
                "fix": fix,
                "split": False,
                "date": date
            }
        )
    elif name == "mix-cifar10-imagenet" and fix == "none":
        metadata = EasyDict(
            {
                "image_size": 32,
                "num_classes": 10,
                "train_images": num_train_baseline,
                "val_images": 10000,
                "color_number": num_train_baseline,
                "gray_number": 0,
                "num_channels": 3,
                "fix": "none",
                "split": False,
                "date": date
            }
        )
    elif name == "mnist-full":
        metadata = EasyDict(
            {
                "image_size": 28,
                "num_classes": 10, # account for full dataset
                "train_images": 60000,
                "val_images": 10000,
                "num_channels": 1,
                "date": date,
                "flip_left": flip_left,
                "flip_right": flip_right,
                "split": False
            }
        )
    elif name == "mnist-subset":
        metadata = EasyDict(
            {
                "image_size": 28,
                "num_classes": 7, # only consider subset of flipped digits
                "train_images": 60000,
                "val_images": 10000,
                "num_channels": 1,
                "date": date,
                "flip_left": flip_left,
                "flip_right": flip_right,
                "split": False
            }
        )
    elif name == "cifar-superclass":
        metadata = EasyDict(
            {
                "image_size": 32,
                "num_classes": 5,
                "train_images": 25000,
                "val_images": 10000,
                "num_channels": 3,
                "front_ratio": front_ratio,
                "back_ratio": back_ratio,
                "date": date,
                "split": False,
                "split_type": semantic_group
            }
        )
    elif name == "celeba":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_attributes": 40,
                "num_classes": 8,
                "female_ratio": female_ratio,
                "male_ratio": male_ratio,
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
    elif name == "fairface":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": 6,
                "female_ratio": female_ratio,
                "male_ratio": male_ratio,
                "date": date,
                "train_images": 30000,
                "val_images": 9745,
                "num_channels": 3,
            }
        )
    elif name == "fairface-base":
        metadata = EasyDict(
            {
                "image_size": 64,
                "train_images": 86744,
                "val_images": 9745,
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
    elif name == "cifar-imagenet":
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = CIFAR_ImageNet(
            split = "train",
            transform=transform_train,
            target_transform=None,
            num_cifar = metadata.num_cifar,
            num_imagenet = metadata.num_imagenet,
            init = False,
            domain_label = False,
            date = metadata.date
        )
    elif name == "cifar-imagenet-check":
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = CIFAR_ImageNet(
            split = "train",
            transform=transform_train,
            target_transform=None,
            num_cifar = metadata.num_cifar,
            num_imagenet = metadata.num_imagenet,
            init = False,
            domain_label = True,
            date = metadata.date
        )
    
    elif name == "mnist-full":
        transform_left = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    metadata.image_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                ),
                transforms.ToTensor(),
            ]
        )
        transform_right = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomResizedCrop(
                    metadata.image_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                ),
                transforms.ToTensor(),
            ]
        )
        train_set = MNIST_FLIP(
            root = os.path.join(data_dir, "mnist"),
            train = True,
            transform_left = transform_left,
            transform_right = transform_right,
            target_transform = None,
            download = True,
            date = metadata.date,
            split = False,
            ratio_left = metadata.flip_left,
            ratio_right = metadata.flip_right,
        )
    
    elif name == "mnist-subset":
        transform_left = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    metadata.image_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                ),
                transforms.ToTensor(),
            ]
        )
        transform_right = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomResizedCrop(
                    metadata.image_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                ),
                transforms.ToTensor(),
            ]
        )
        train_set = MNIST_FLIP(
            root = os.path.join(data_dir, "mnist"),
            train = True,
            transform_left = transform_left,
            transform_right = transform_right,
            target_transform = None,
            download = True,
            date = metadata.date,
            split = False,
            ratio_left = metadata.flip_left,
            ratio_right = metadata.flip_right,
        )
    
    elif name == "cifar-superclass":
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = CIFAR_SuperClass(
            root = os.path.join(data_dir, "cifar10"),
            transform = transform_train,
            target_transform = None,
            train = True,
            download = False,
            front_ratio = metadata.front_ratio,
            back_ratio = metadata.back_ratio,
            split = False,
            split_type = metadata.split_type,
            date = metadata.date,
        )

    elif name == "celeba":
        # celebA has a large number of images, avoiding randomcropping.
        transform_train = transforms.Compose(
            [
                transforms.Resize(metadata.image_size),
                transforms.CenterCrop(metadata.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = CelebA_AttrCond(
            split="train",
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
    elif name == "fairface":
        trainsform_train = transforms.Compose(
            [
                transforms.Resize(metadata.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        )
        train_set = FairFace_Gender(
            split="train", 
            transform=trainsform_train, 
            f_ratio=metadata.female_ratio, 
            m_ratio=metadata.male_ratio, 
            date=metadata.date
        )
    elif name == "fairface-base":
        trainsform_train = transforms.Compose(
            [
                transforms.Resize(metadata.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        )
        train_set = FairFace_Base(
            split="train",
            transform=trainsform_train
        )
    else:
        raise ValueError(f"{name} dataset nor supported!")
    return train_set


# TODO: Add datasets imagenette/birds/svhn etc etc.
def get_domain_dataset(name, data_dir, metadata):
    """
    Return a dataset with the current name. We only support two datasets with
    their fixed image resolutions. One can easily add additional datasets here.

    This is for binary domain classification only. So not use any data augmentation.
    """
    if name == "cifar10":
        transform_train = transforms.Compose(
            [
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
    elif name == "cifar-imagenet":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])
            ]
        )
        train_set = CIFAR_ImageNet(
            split = "train",
            transform=transform_train,
            target_transform=None,
            num_cifar = metadata.num_cifar,
            num_imagenet = metadata.num_imagenet,
            init = False,
            date = metadata.date
        )
    
    elif name == "mnist-full":
        transform_left = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    metadata.image_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                ),
                transforms.ToTensor(),
            ]
        )
        transform_right = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomResizedCrop(
                    metadata.image_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                ),
                transforms.ToTensor(),
            ]
        )
        train_set = MNIST_FLIP(
            root = os.path.join(data_dir, "mnist"),
            train = True,
            transform_left = transform_left,
            transform_right = transform_right,
            target_transform = None,
            download = True,
            date = metadata.date,
            split = False,
            ratio_left = metadata.flip_left,
            ratio_right = metadata.flip_right,
        )
    
    elif name == "mnist-subset":
        transform_left = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    metadata.image_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                ),
                transforms.ToTensor(),
            ]
        )
        transform_right = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomResizedCrop(
                    metadata.image_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                ),
                transforms.ToTensor(),
            ]
        )
        train_set = MNIST_FLIP(
            root = os.path.join(data_dir, "mnist"),
            train = True,
            transform_left = transform_left,
            transform_right = transform_right,
            target_transform = None,
            download = True,
            date = metadata.date,
            split = False,
            ratio_left = metadata.flip_left,
            ratio_right = metadata.flip_right,
        )
    
    elif name == "cifar-superclass":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        train_set = CIFAR_SuperClass(
            root = os.path.join(data_dir, "cifar10"),
            transform = transform_train,
            target_transform = None,
            train = True,
            download = False,
            front_ratio = metadata.front_ratio,
            back_ratio = metadata.back_ratio,
            split = False,
            split_type = metadata.split_type,
            date = metadata.date,
        )

    elif name == "celeba":
        # celebA has a large number of images, avoiding randomcropping.
        transform_train = transforms.Compose(
            [
                transforms.Resize(metadata.image_size),
                transforms.CenterCrop(metadata.image_size),
                transforms.ToTensor(),
            ]
        )
        train_set = CelebA_AttrCond(
            split="train",
            target_type="attr",
            transform=transform_train,
        )
    elif name == "celeba-hq":
        transform_train = transforms.Compose(
            [
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
    elif name == "fairface":
        trainsform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalized by imagenet mean std
            ]
        )
        train_set = FairFace_Gender(
            split="train", 
            transform=trainsform_train, 
            f_ratio=metadata.female_ratio, 
            m_ratio=metadata.male_ratio, 
            date=metadata.date
        )
    else:
        raise ValueError(f"{name} dataset nor supported!")
    return train_set
