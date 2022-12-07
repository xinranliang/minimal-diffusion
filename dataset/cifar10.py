import os
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


def main():
    for split in ["train", "test"]:
        out_dir = f"/home/xinranliang/projects/minimal-diffusion/datasets/cifar_raw/cifar_{split}"
        if os.path.exists(out_dir):
            print(f"skipping split {split} since {out_dir} already exists.")
            continue

        print("downloading...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = torchvision.datasets.CIFAR10(
                root=tmp_dir, train=split == "train", download=True
            )

        print("dumping images...")
        os.makedirs(out_dir, exist_ok=True)
        for i in tqdm(range(len(dataset))):
            image, label = dataset[i]
            os.makedirs(os.path.join(out_dir, CLASSES[label]), exist_ok=True)
            filename = os.path.join(out_dir, f"{CLASSES[label]}/{i:05d}.png")
            image.save(filename)


class CIFAR10_Custom(datasets.CIFAR10):
    def __init__(
        self,
        root,
        train,
        transform,
        target_transform,
        download,
        color_ratio, 
        grayscale_ratio
    ):
        super().__init__(root, train, transform, target_transform, download=download)

        self.num_classes = 10 # default 10 classes
        assert color_ratio + grayscale_ratio == 1.0
        self.color_ratio = color_ratio
        self.grayscale_ratio = grayscale_ratio

        # randomly sample color v.s. grayscale classes
        self.color_index = np.random.choice(len(self.data), int(len(self.data) * self.color_ratio), replace=False)
        self.grayscale_index = [no_idx for no_idx in range(len(self.data)) if no_idx not in self.color_index]

        assert len(self.color_index) == int(len(self.data) * self.color_ratio)
        assert len(self.grayscale_index) == int(len(self.data) * self.grayscale_ratio)
    

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
    # main()

    transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    # construct custom cifar10 dataset
    cifar10_custom = CIFAR10_Custom("/home/xinranliang/projects/minimal-diffusion/datasets/cifar_color", True, transform_train, None, False, 0.05, 0.95)

    # build data loader
    data_loader = DataLoader(cifar10_custom, batch_size=4, shuffle=True, num_workers=4)

    image, label = next(iter(data_loader))
    