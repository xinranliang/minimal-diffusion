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


flip_classes = (
    2, 3, 4, 5, 6, 7, 9
)

def generate_mnist_index(date):
    transform_left = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    28, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                ),
                transforms.ToTensor(),
            ]
    )
    transform_right = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomResizedCrop(
                    28, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                ),
                transforms.ToTensor(),
            ]
    )

    for left, right in zip([0.9, 0.7, 0.5, 0.3, 0.1], [0.1, 0.3, 0.5, 0.7, 0.9]):
        mydataset = MNIST_FLIP(
            root = "/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/mnist",
            transform_left = transform_left,
            transform_right = transform_right,
            target_transform = None,
            date = date,
            ratio_left = left,
            ratio_right = right,
            split=True
        )

def get_mnist_index(date):
    transform_left = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    28, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                ),
                transforms.ToTensor(),
            ]
    )
    transform_right = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomResizedCrop(
                    28, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                ),
                transforms.ToTensor(),
            ]
    )
    mydataset = MNIST_FLIP(
        root = "/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/mnist",
        transform_left = transform_left,
        transform_right = transform_right,
        target_transform = None,
        date = date,
        ratio_left = 0.5,
        ratio_right = 0.5,
        split=False
    )
    mydataloader = DataLoader(mydataset, batch_size=1, shuffle=True)
    os.makedirs(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/mnist", "MNIST_FLIP/test", date), exist_ok=True)
    for index, (image, label) in enumerate(mydataloader):
        image = (image * 255.0).numpy().astype(np.uint8).reshape((-1, 28, 28)).transpose((1, 2, 0))
        cv2.imwrite(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/mnist", "MNIST_FLIP/test", date, f"index_{index}.png"), image)
        input("press enter to continue")



class MNIST_FLIP(datasets.MNIST):
    def __init__(
        self, 
        root,
        transform_left,
        transform_right,
        target_transform,
        date,
        ratio_left,
        ratio_right,
        split=False,
        train=True,
        download=True,
    ):
        super().__init__(root, train, None, target_transform, download=download)

        self.transform_left = transform_left
        self.transform_right = transform_right
        self.ratio_left = ratio_left
        self.ratio_right = ratio_right
        assert self.ratio_left + self.ratio_right == 1.0

        if split:
            self.index_by_label = {
                2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 9: [],
            }
            self.index_left = {
                2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 9: [],
            }
            self.index_right = {
                2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 9: [],
            }

            # all indexes by classes
            for index in range(len(self.data)):
                label = int(self.targets[index])
                if label in self.index_by_label.keys():
                    self.index_by_label[label].append(index)
            
            # sample by class
            for class_label in flip_classes:
                num_data = len(self.index_by_label[class_label])
                print("class {} has {} images".format(class_label, num_data))
                self.index_by_label[class_label] = np.array(self.index_by_label[class_label], dtype=int)
                self.index_left[class_label] = np.random.choice(self.index_by_label[class_label], size=int(self.ratio_left * num_data), replace=False)
                self.index_right[class_label] = [no_idx for no_idx in self.index_by_label[class_label] if no_idx not in self.index_left[class_label]]
                self.index_right[class_label] = np.array(self.index_right[class_label], dtype=int)
                assert len(self.index_left[class_label]) == int(self.ratio_left * num_data)
                assert len(self.index_right[class_label]) == num_data - int(self.ratio_left * num_data)
            
            # save
            idx_dict = {"index_left": self.index_left, "index_right": self.index_right}
            file_path = os.path.join(root, "MNIST_FLIP", "flip_index_split", date, "left{}_right{}_split.pkl".format(self.ratio_left, self.ratio_right))
            with open(file_path, "wb") as f:
                pickle.dump(idx_dict, f)
        
        else:
            file_path = os.path.join(root, "MNIST_FLIP", "flip_index_split", date, "left{}_right{}_split.pkl".format(self.ratio_left, self.ratio_right))
            with open(file_path, "rb") as f:
                file_load = pickle.load(f)
            print("Loading left/right horizontal flip split from file path {}".format(file_path))
            self.index_left = file_load["index_left"]
            self.index_right = file_load["index_right"]
    

    def __getitem__(self, index):
        image, target = super().__getitem__(index)

        if target in flip_classes:
            if index in self.index_left[target]:
                # print(f"Left index {index} of class {target}, no flipping")
                image = self.transform_left(image)
            elif index in self.index_right[target]:
                # print(f"Right index {index} of class {target}, flipping")
                image = self.transform_right(image)
            else:
                raise ValueError("an index should either in left or right flip")
        else:
            # print(f"symmetric class of {target} at index {index}")
            image = self.transform_left(image)
        
        return image, target



if __name__ == "__main__":
    # generate_mnist_index("2023-04-08")
    # generate_mnist_index("2023-04-09")
    get_mnist_index("2023-04-08")