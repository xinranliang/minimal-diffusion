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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

def recover_cifar10():
    root = "/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet"
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    sets = ["test", "valid"]
    for class_name in classes:
        for set_name in sets:
            filenames_source = glob.glob('{}/{}/{}/cifar10-train-*.png'.format(root, set_name, class_name))
            for source_file in filenames_source:
                target_file = source_file.replace(set_name, "train")
                copyfile(source_file, target_file)

def recover_imagenet():
    root = "/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet"
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    sets = ["test", "valid"]
    for class_name in classes:
        for set_name in sets:
            filenames_source = glob.glob('{}/{}/{}/*.png'.format(root, set_name, class_name))
            for source_file in filenames_source:
                # exclude cifar data
                if "cifar10" not in source_file.split("/")[-1]:
                    source_image = Image.open(source_file)
                    target_file = source_file.replace(set_name, "train")
                    source_image.save(target_file)
                    # copyfile(source_file, target_file)

def generate_fix_cifar10_other_index(fix_num, other_num, other_dataset, date):
    # base domain dataset - CIFAR10 color
    if fix_num < 50000:
        fix_index = np.random.choice(50000, size=fix_num, replace=False)
    elif fix_num == 50000:
        fix_index = range(50000)
    assert len(fix_index) == fix_num

    if other_dataset == "imagenet":
        full_dataset = datasets.ImageFolder(
                            root = "/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/train",
                            transform=transforms.Compose([transforms.ToTensor(),]),
                            target_transform=None
                        )
        # List of (image_path, class_index) tuples
        cifar10_paths = []
        imagenet_paths = []

        for image_path, image_label in full_dataset.imgs:
            image_name = image_path.split("/")[-1]
            # extract cifar10 subset
            if "cifar10" in image_name:
                image_index = image_name.split(".")[0].split("-")[-1]
                if int(image_index) in fix_index:
                    cifar10_paths.append((image_path, image_label))
            # extract imagenet subset
            elif "cifar10" not in image_name:
                imagenet_paths.append((image_path, image_label))
        
        assert len(cifar10_paths) == fix_num
        # sample from imagenet subset
        assert other_num <= len(imagenet_paths), "number of required samples {} is more than number of available samples {}".format(other_num, len(imagenet_paths))
        if other_num < len(imagenet_paths):
            imagenet_paths = random.sample(imagenet_paths, other_num)
        assert len(imagenet_paths) == other_num
        # concatenate and random shuffle
        img_paths = cifar10_paths + imagenet_paths
        assert len(img_paths) == fix_num + other_num
        random.shuffle(img_paths)

        # save
        with open(
            os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/index_split", date, "cifar{}_imagenet{}_index.pkl".format(fix_num, other_num)), "wb"
        ) as f:
            pickle.dump(img_paths, f)

    else:
        raise NotImplementedError


def generate_cifar10_imagenet_index(date):
    generate_fix_cifar10_other_index(25000, 0, "imagenet", date)

    other_num = [0, 25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000]
    for number in other_num:
        generate_fix_cifar10_other_index(50000, number, "imagenet", date)

def generate_imagenet_index(date):
    generate_fix_cifar10_other_index(0, 50000, "imagenet", date)


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


class CIFAR10_Other(Dataset):
    def __init__(
        self,
        root, # cifar10 root
        train, # train split, default True
        transform, # data augmentation
        target_transform, # default None
        fix_num, # base dataset size
        other_num, # other dataset size
        other_name, # name of other dataset
        date, # date of experiment to handle multiple runs
    ):
        super().__init__()

        # number and resolution
        self.image_size = 32
        self.fix_num = fix_num
        self.other_num = other_num

        self.other_name = other_name
        if self.other_name == "cifar100":
            self.fix_dataset = datasets.CIFAR10(
                root="/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10",
                train=train,
                transform=transform,
                target_transform=target_transform,
                download=False
            )
            self.other_dataset = datasets.CIFAR100(
                root = "/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar100",
                train=train,
                transform=transform,
                target_transform=target_transform,
                download=False
            )
        elif self.other_name == "mnist":
            self.fix_dataset = datasets.CIFAR10(
                root="/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10",
                train=train,
                transform=transform,
                target_transform=target_transform,
                download=False
            )
            transform_mnist = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        self.image_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                    ),
                    transforms.ToTensor(),
                ]
            )
            self.other_dataset = datasets.MNIST(
                root = "/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/mnist",
                train=train,
                transform=transform_mnist,
                target_transform=None,
                download=True
            )
        elif self.other_name == "imagenet":
            self.full_dataset = datasets.ImageFolder(
                root = "/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/train",
                transform=transform,
                target_transform=target_transform
            )
            # load index file
            file_path = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/index_split", date, "cifar{}_imagenet{}_index.pkl".format(fix_num, other_num))
            with open(file_path, "rb") as f:
                self.full_dataset.samples = pickle.load(f)
            print("Loading cifar10/imagenet index split from file path {}".format(file_path))

        else:
            raise NotImplementedError
    
    def __len__(self):
        return self.fix_num + self.other_num
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.other_name == "imagenet":
            return self.full_dataset.__getitem__(index)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    # recover_cifar10()
    # recover_imagenet()
    # generate index
    # imagenet
    # generate_cifar10_imagenet_index(date="2023-04-06")
    # generate_cifar10_imagenet_index(date="2023-04-07")
    generate_imagenet_index(date="2023-04-06")
    generate_imagenet_index(date="2023-04-07")