import os
import csv
from easydict import EasyDict
from typing import Optional
from collections import namedtuple, defaultdict

from tqdm.auto import tqdm

import numpy as np
import random
from PIL import Image
import cv2
import scipy, scipy.io
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

root_path = "/n/fs/visualai-scr/Data/CelebA"

gender_independent_attrs = ["Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Chubby", "Bags_Under_Eyes", "Eyeglasses", "Gray_Hair", "High_Cheekbones", "Mouth_Slightly_Open", "Narrow_Eyes", "Smiling", "Wearing_Earrings", "Wearing_Hat"]
gender_dependent_attrs = ["Arched_Eyebrows", "Attractive", "Bushy_Eyebrows", "Pointy_Nose", "Receding_Hairline", "Young"]
all_attrs = sorted(gender_dependent_attrs + gender_independent_attrs)

my_attrs = ["Attractive", "Mouth_Slightly_Open"]

class CelebA_AttrCond(datasets.CelebA):
    def __init__(
        self,
        split,
        target_type,
        transform,
        target_transform=None,
        my_attrs=my_attrs,
        root=root_path,
        download=False
    ):
        super().__init__(root, split, target_type, transform, target_transform, download)

        self.my_attrs = my_attrs
        self.group_attr_idx = [self.attr_names.index(my_attr) for my_attr in self.my_attrs]
        # self.group_attr_idx.append(self.attr_names.index("Male"))
        self.group_attr_idx = torch.tensor(self.group_attr_idx, dtype=torch.long)
        self.num_classes = 2 ** len(self.my_attrs)
    
    def __getitem__(self, index):
        image, attr_target = super().__getitem__(index)
        # attribute target shape = (40,)
        myattr_label = self.get_target(attr_target)
        gender_label = torch.select(attr_target, 0, self.attr_names.index("Male")).item()
        return image, myattr_label, gender_label
    
    def get_target(self, attr_target):
        myattr_target = torch.gather(attr_target, 0, self.group_attr_idx)
        for i in range(self.num_classes):
            if myattr_target[0] == 0 and myattr_target[1] == 0 and myattr_target[2] == 0:
                return int(0)
            elif myattr_target[0] == 1 and myattr_target[1] == 0 and myattr_target[2] == 0:
                return int(1)
            elif myattr_target[0] == 0 and myattr_target[1] == 1 and myattr_target[2] == 0:
                return int(2)
            elif myattr_target[0] == 0 and myattr_target[1] == 0 and myattr_target[2] == 1:
                return int(3)
            elif myattr_target[0] == 1 and myattr_target[1] == 1 and myattr_target[2] == 0:
                return int(4)
            elif myattr_target[0] == 1 and myattr_target[1] == 0 and myattr_target[2] == 1:
                return int(5)
            elif myattr_target[0] == 0 and myattr_target[1] == 1 and myattr_target[2] == 1:
                return int(6)
            elif myattr_target[0] == 1 and myattr_target[1] == 1 and myattr_target[2] == 1:
                return int(7)
            else:
                raise ValueError("No class exists in this combination!")


def get_attribute_distribution():
    celeba = datasets.CelebA(
        root=root_path,
        split="train",
        target_type="attr",
        transform=transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
        target_transform=None,
        download=False
    )
    attr_dist = defaultdict(int)
    for i in range(2):
        attr_dist[i] = np.zeros(40, dtype=int)
    for image, target in celeba:
        zero_index = torch.where(target == 0)[0].numpy()
        one_index = torch.where(target == 1)[0].numpy()
        attr_dist[0][zero_index] += 1
        attr_dist[1][one_index] += 1
    print(attr_dist)
    # plot
    fig, ax = plt.subplots()
    low = np.zeros(40)
    for k, v in attr_dist.items():
        ax.bar(celeba.attr_names[:-1], v, 0.5, label=k, bottom=low)
        low += v
    ax.set_title("Number of 0/1 samples with each attribute")
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
    ax.legend()
    plt.savefig("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/celeba/attr_dist.png", dpi=300)
    plt.savefig("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/celeba/attr_dist.pdf", dpi=300)


def get_gender_per_label_dist():
    print(all_attrs)
    os.makedirs("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/celeba/gender_dist", exist_ok=True)
    for i in range(len(all_attrs)):
        for j in range(i+1, len(all_attrs)):
            for k in range(j+1, len(all_attrs)):
                print("current list of attributes: [{}, {}, {}]".format(all_attrs[i], all_attrs[j], all_attrs[k]))
                
                myceleba = CelebA_AttrCond(
                    root=root_path,
                    split="train",
                    target_type="attr",
                    my_attrs=[all_attrs[i], all_attrs[j], all_attrs[k]],
                    transform=transforms.Compose(
                        [
                            transforms.Resize(64),
                            transforms.CenterCrop(64),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ]
                    ),
                    target_transform=None,
                    download=False
                )

                count_labels = defaultdict(int)
                for n in range(2):
                    count_labels[n] = np.zeros(myceleba.num_classes, dtype=int)
                for image, attr_label, gender_label in myceleba:
                    count_labels[gender_label][attr_label] += 1
                # plot
                fig, ax = plt.subplots()
                low = np.zeros(myceleba.num_classes)
                for k, v in count_labels.items():
                    ax.bar(np.arange(myceleba.num_classes), v, 0.5, label=k, bottom=low)
                    low += v
                ax.set_title(f"Number of Female/Male samples using Attribute [{myceleba.my_attrs[0]}, {myceleba.my_attrs[1]}, {myceleba.my_attrs[2]}]", fontsize=8)
                ax.legend()
                plt.savefig(f"/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/celeba/gender_dist/{myceleba.my_attrs[0]}+{myceleba.my_attrs[1]}+{myceleba.my_attrs[2]}.png", dpi=300)
                plt.savefig(f"/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/celeba/gender_dist/{myceleba.my_attrs[0]}+{myceleba.my_attrs[1]}+{myceleba.my_attrs[2]}.pdf", dpi=300)
                plt.close()


def get_all_gender_dist():
    celeba = datasets.CelebA(
        root=root_path,
        split="train",
        target_type="attr",
        transform=transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
        target_transform=None,
        download=False
    )
    count_gender = defaultdict(int)
    for image, target in celeba:
        gender_label = torch.select(target, 0, celeba.attr_names.index("Male")).item()
        count_gender[gender_label] += 1
    # plot
    fig, ax = plt.subplots()
    x_label = ["Female", "Male"]
    y_value = [count_gender[0], count_gender[1]]
    ax.bar(x_label, y_value)
    ax.set_ylabel('number of samples')
    ax.set_title('Gender distribution in CelebA-Train')
    plt.savefig("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/celeba/all_gender_dist.png", dpi=300)
    plt.savefig("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/celeba/all_gender_dist.pdf", dpi=300)

if __name__ == "__main__":
    # get_attribute_distribution()
    get_gender_per_label_dist()
    # get_all_gender_dist()