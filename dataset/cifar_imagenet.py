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
NUM_TOTAL = 50000

def split_cifar_imagenet_index(date):
    transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    
    for n_cifar, n_imgnet in zip([0.1, 0.3, 0.5, 0.7, 0.9], [0.9, 0.7, 0.5, 0.3, 0.1]):
        assert n_cifar + n_imgnet == 1.0
        my_cifar_imgnet = CIFAR_ImageNet(
            split="train",
            transform=transform_train,
            target_transform=None,
            num_cifar=n_cifar,
            num_imagenet=n_imgnet,
            init=True,
            date=date
        )

class CIFAR_ImageNet(Dataset):
    def __init__(
        self,
        split, # train split, default True
        transform, # data augmentation
        target_transform, # default None
        num_cifar, # base dataset size
        num_imagenet, # other dataset size
        init, # whether initialize subsample index
        date, # date of experiment to handle multiple runs
    ):
        super().__init__()

        # number and resolution, need to input proportion is fine
        self.image_size = 32
        self.num_cifar = int(num_cifar * NUM_TOTAL)
        self.num_imagenet = int(num_imagenet * NUM_TOTAL)

        assert split in ["train", "valid", "test"]
        self.full_dataset = datasets.ImageFolder(
            root = f"/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/{split}",
            transform=transform,
            target_transform=target_transform
        )

        if init:
            # List of (image_path, class_index) tuples ordered by class label
            cifar_paths = {index: [] for index in range(len(CLASSES))}
            imagenet_paths = {index: [] for index in range(len(CLASSES))}

            for image_path, image_label in self.full_dataset.imgs:
                image_name = image_path.split("/")[-1]
                # extract cifar10 subset
                if "cifar10" in image_name:
                    # image_index = image_name.split(".")[0].split("-")[-1]
                    cifar_paths[image_label].append((image_path, image_label))
                # extract imagenet subset
                else:
                    imagenet_paths[image_label].append((image_path, image_label))

            # sample from cifar/imagenet subset respectively
            num_cifar_per_class = self.num_cifar // len(CLASSES)
            num_imagenet_per_class = self.num_imagenet // len(CLASSES)
            sub_cifar_paths = {}
            sub_imagenet_paths = {}
            for index in range(len(CLASSES)):
                sub_cifar_paths[index] = random.sample(cifar_paths[index], num_cifar_per_class)
                sub_imagenet_paths[index] = random.sample(imagenet_paths[index], num_imagenet_per_class)

            concat_cifar_paths = []
            concat_imagenet_paths = []
            for index in range(len(CLASSES)):
                concat_cifar_paths += sub_cifar_paths[index]
                print(f"number of CIFAR samples with class label {CLASSES[index]} is {len(sub_cifar_paths[index])}")
                concat_imagenet_paths += sub_imagenet_paths[index]
                print(f"number of ImageNet samples with class label {CLASSES[index]} is {len(sub_imagenet_paths[index])}")
            
            # assert correct length
            assert len(concat_cifar_paths) == self.num_cifar
            assert len(concat_imagenet_paths) == self.num_imagenet
            # concatenate and random shuffle
            img_paths = concat_cifar_paths + concat_imagenet_paths
            assert len(img_paths) == self.num_cifar + self.num_imagenet
            random.shuffle(img_paths)

            file_path = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/index_split", date)
            os.makedirs(file_path, exist_ok=True)
            file_path = os.path.join(file_path, "cifar{}_imagenet{}_index.pkl".format(self.num_cifar, self.num_imagenet))
            with open(file_path, "wb") as f:
                pickle.dump(img_paths, f)
        
        else:
            # load index file
            file_path = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/index_split", date, "cifar{}_imagenet{}_index.pkl".format(self.num_cifar, self.num_imagenet))
            with open(file_path, "rb") as f:
                self.full_dataset.samples = pickle.load(f)
            print("Loading cifar10/imagenet index split from file path {}".format(file_path))

            # assert correct length
            assert len(self.full_dataset.samples) == self.num_cifar + self.num_imagenet
    

    def __len__(self):
        return self.num_cifar + self.num_imagenet
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        return self.full_dataset.__getitem__(index)


if __name__ == "__main__":
    split_cifar_imagenet_index(date="2023-07-31")
    split_cifar_imagenet_index(date="2023-08-01")