import os
import csv
from easydict import EasyDict
from typing import Optional
from collections import namedtuple

from tqdm.auto import tqdm

import numpy as np
import random
from PIL import Image
import cv2
import scipy, scipy.io

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


class CelebA_Custom(Dataset):
    def __init__(
        self,
        root,
        target_type,
        transform,
        target_transform=None,
    ):
        super().__init__()

        # directory
        self.root = root 

        # deal with attribute
        attribute = self._load_csv("list_attr_celeba.csv")
        self.attribute = attribute["data"]
        self.attribute_names = attribute["header"]

        self.filename = attribute["index"]

        # data transform
        self.transform = transform
        self.target_transform = target_transform
    
    def _load_csv(
        self,
        filename: str
    ):

        attribute = {
            "header": [], # list of attribute names in string
            "index": [], # list of image filename in string
            "data": [], # list of binary assignment of attribute
        }

        with open(os.path.join(self.root, filename)) as csv_file:
            data = csv.reader(csv_file, delimiter=" ", skipinitialspace=True)

            headers = next(data)[0].split(",")
            for element in headers[1: ]:
                attribute["header"].append(element)

            for row in data:
                attr_list = row[0].split(",")
                attribute["index"].append(
                    os.path.join(self.root, "img_align_celeba", attr_list[0])
                )
                attr_vec = np.zeros((40,), dtype=np.int32)
                for assign_index in range(len(attr_list[1: ])):
                    attr_vec[assign_index] = int(attr_list[1 + assign_index])
                # map from {-1, 1} to {0, 1}
                attr_vec = np.floor_divide(attr_vec + 1, 2)

                attribute["data"].append(torch.tensor(attr_vec))

        return attribute
    
    def __len__(self):
        return len(self.filename)
    
    def __getitem__(self, index):
        image = Image.open(self.filename[index])

        target = self.attribute[index]

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target


if __name__ == "__main__":
    transform_train = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    # construct custom cifar10 dataset
    celeba_dataset = CelebA_Custom(
        root="/home/xinranliang/projects/minimal-diffusion/datasets/celebA", 
        target_type="attr",
        transform=transform_train,
    )

    # build data loader
    data_loader = DataLoader(celeba_dataset, batch_size=4, shuffle=True, num_workers=4)

    image, attribute = next(iter(data_loader))
    print(image.shape)
    print(attribute)