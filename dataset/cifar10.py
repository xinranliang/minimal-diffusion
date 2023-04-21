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

SUPER_CLASSES = {
    0: [0, 8], # "airplane" + "ship"
    1: [1, 9], # "automobile" + "truck"
    2: [2, 6], # "bird" + "frog"
    3: [3, 5], # "cat" + "dog"
    4: [4, 7], # "deer" + "horse"
}
SUPER_CLASSES_INVERSE = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 3, 6: 2, 7: 4, 8: 0, 9: 1
}


def generate_index_cifar10():
    transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    for color, gray in zip([0.95, 0.9, 0.7, 0.5, 0.3, 0.1, 0.05], [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]):
        cifar10 = CIFAR10_ColorGray(
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
        cifar10 = CIFAR10_ColorGray(
            root="/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10",
            train=True,
            transform=transform_train,
            target_transform=None,
            download=False,
            color_ratio=color,
            grayscale_ratio=gray,
            split=False # whether we're performing one-off splitting
        )


def generate_fixgroup_cifar10():
    transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    
    fix_num = 15000
    # 0%, 20%, 40%, 60%, 80%, 100%, 200%, 400%, 800%, 1600%
    options = [0, 5000, 10000, 15000, 20000, 25000, 30000]

    for number in options:
        cifar10 = CIFAR10_FixGroup(
                root="/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10",
                train=True,
                transform=transform_train,
                target_transform=None,
                download=False,
                fix="color",
                color_number=fix_num,
                gray_number=number,
                split=True # whether we're performing one-off splitting
        )
        print("Construct dataset with {} color and {} gray images".format(fix_num, number))
        print("Color index has length {}".format(len(cifar10.color_index)))
        print("Gray index has length {}".format(len(cifar10.gray_index)))

def check_fixgroup_cifar10():
    transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    
    fix_num = 15000
    options = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000]

    for number in options:
        cifar10 = CIFAR10_FixGroup(
                root="/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10",
                train=True,
                transform=transform_train,
                target_transform=None,
                download=False,
                color_number=fix_num,
                gray_number=number,
                split=False # whether we're performing one-off splitting
        )

        print("Construct dataset with {} color and {} gray images".format(fix_num, number))
        print("Color index has length {}".format(len(cifar10.color_index)))
        print("Gray index has length {}".format(len(cifar10.gray_index)))
        image, label = cifar10.__getitem__(200)
    
def split_by_class(date):
    transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    front_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    back_ratios = [0.9, 0.7, 0.5, 0.3, 0.1]
    for front_ratio, back_ratio in zip(front_ratios, back_ratios):
        mydata = CIFAR_SuperClass(
            root="/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10",
            train=True,
            transform=transform_train,
            target_transform=None,
            download=False,
            split=True,
            date=date,
            front_ratio=front_ratio,
            back_ratio=back_ratio
        )

def check_by_class(date):
    transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    front_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    back_ratios = [0.9, 0.7, 0.5, 0.3, 0.1]
    for front_ratio, back_ratio in zip(front_ratios, back_ratios):
        mydata = CIFAR_SuperClass(
            root="/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10",
            train=True,
            transform=transform_train,
            target_transform=None,
            download=False,
            split=False,
            date=date,
            front_ratio=front_ratio,
            back_ratio=back_ratio
        )


class CIFAR10_ColorGray(datasets.CIFAR10):
    def __init__(
        self,
        root,
        train,
        transform,
        target_transform,
        download,
        color_ratio = None, 
        grayscale_ratio = None,
        split = False,
        date = ""
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

            file_path = os.path.join(root, "color_gray_split", date, "color{}_gray{}_split.pkl".format(self.color_ratio, self.grayscale_ratio))
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
            split_file_path = os.path.join(root, "color_gray_split", date, "color{}_gray{}_split.pkl".format(self.color_ratio, self.grayscale_ratio))
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
            img = rgb_to_gray(img)
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


class CIFAR10_FixGroup(datasets.CIFAR10):
    def __init__(
        self,
        root,
        train,
        transform,
        target_transform,
        download,
        fix,
        color_number = None, 
        gray_number = None,
        split = False,
        date = ""
    ):
        super().__init__(root, train, transform, target_transform, download=download)

        self.num_classes = 10 # default 10 classes
        self.color_number = color_number
        self.gray_number = gray_number
        self.fix = fix

        # get color index and gray index
        if split == True and self.color_number != 0 and self.gray_number != 0:
            if self.fix == "color":
                if self.color_number == 2500:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.05_gray0.95_split.pkl")
                elif self.color_number == 5000:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.1_gray0.9_split.pkl")
                elif self.color_number == 15000:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.3_gray0.7_split.pkl")
                elif self.color_number == 25000:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.5_gray0.5_split.pkl")
                elif self.color_number == 35000:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.7_gray0.3_split.pkl")
                elif self.color_number == 45000:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.9_gray0.1_split.pkl")
                elif self.color_number == 47500:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.95_gray0.05_split.pkl")
                else: 
                    raise NotImplementedError
            elif self.fix == "gray":
                if self.gray_number == 47500:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.05_gray0.95_split.pkl")
                elif self.gray_number == 45000:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.1_gray0.9_split.pkl")
                elif self.gray_number == 35000:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.3_gray0.7_split.pkl")
                elif self.gray_number == 25000:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.5_gray0.5_split.pkl")
                elif self.gray_number == 15000:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.7_gray0.3_split.pkl")
                elif self.gray_number == 5000:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.9_gray0.1_split.pkl")
                elif self.gray_number == 2500:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.95_gray0.05_split.pkl")
                else: 
                    raise NotImplementedError
            else:
                raise NotImplementedError

            with open(split_file_path, "rb") as f:
                file_load = pickle.load(f)
            self.color_index = file_load["color_index"]
            self.gray_index = file_load["gray_index"]
            if self.fix == "color":
                assert self.gray_number <= len(self.gray_index)
                self.gray_index = np.random.choice(self.gray_index, size=self.gray_number, replace=False)
            elif self.fix == "gray":
                assert self.color_number <= len(self.color_index)
                self.color_index = np.random.choice(self.color_index, size=self.color_number, replace=False)
            else:
                raise NotImplementedError

            idx_dict = {"color_index": self.color_index, "gray_index": self.gray_index}

            file_path = os.path.join(root, "color_gray_split", date, "color{}_gray{}_index.pkl".format(self.color_number, self.gray_number))
            with open(file_path, "wb") as f:
                pickle.dump(idx_dict, f)
        
        elif split == True and (self.color_number == 0 or self.gray_number == 0):
            # one of subgroup has no instance
            if self.fix == "color":
                if self.color_number == 2500:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.05_gray0.95_split.pkl")
                elif self.color_number == 5000:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.1_gray0.9_split.pkl")
                elif self.color_number == 15000:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.3_gray0.7_split.pkl")
                elif self.color_number == 25000:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.5_gray0.5_split.pkl")
                elif self.color_number == 35000:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.7_gray0.3_split.pkl")
                elif self.color_number == 45000:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.9_gray0.1_split.pkl")
                elif self.color_number == 47500:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.95_gray0.05_split.pkl")
                else: 
                    raise NotImplementedError
            elif self.fix == "gray":
                if self.gray_number == 47500:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.05_gray0.95_split.pkl")
                elif self.gray_number == 45000:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.1_gray0.9_split.pkl")
                elif self.gray_number == 35000:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.3_gray0.7_split.pkl")
                elif self.gray_number == 25000:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.5_gray0.5_split.pkl")
                elif self.gray_number == 15000:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.7_gray0.3_split.pkl")
                elif self.gray_number == 5000:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.9_gray0.1_split.pkl")
                elif self.gray_number == 2500:
                    split_file_path = os.path.join(root, "color_gray_split", date, "color0.95_gray0.05_split.pkl")
                else: 
                    raise NotImplementedError
            else:
                raise NotImplementedError

            with open(split_file_path, "rb") as f:
                file_load = pickle.load(f)
            if self.fix == "color":
                self.color_index = file_load["color_index"]
                self.gray_index = []
            elif self.fix == "gray":
                self.gray_index = file_load["gray_index"]
                self.color_index = []
            else:
                raise NotImplementedError

            idx_dict = {"color_index": self.color_index, "gray_index": self.gray_index}

            file_path = os.path.join(root, "color_gray_split", date, "color{}_gray{}_index.pkl".format(self.color_number, self.gray_number))
            with open(file_path, "wb") as f:
                pickle.dump(idx_dict, f)

        elif split == False:
            file_path = os.path.join(root, "color_gray_split", date, "color{}_gray{}_index.pkl".format(self.color_number, self.gray_number))
            with open(file_path, "rb") as f:
                file_load = pickle.load(f)
            print("Loading color/gray index from file path {}".format(file_path))
            self.color_index = file_load["color_index"]
            self.gray_index = file_load["gray_index"]
        
        else:
            raise NotImplementedError
        
        # random shuffle all color and gray index
        # both are numpy array (color_number, ) (gray_number, )
        self.image_index = np.concatenate((self.color_index, self.gray_index), axis=0).astype(int)
        random.shuffle(self.image_index)
    

    def __len__(self):
        return self.color_number + self.gray_number
    
    def __getitem__(self, index):
        img_idx = self.image_index[index]
        img, target = self.data[img_idx], self.targets[img_idx]

        # decide color or grayscale
        if img_idx in self.gray_index:
            assert img_idx not in self.color_index
            img = rgb_to_gray(img)
            img = img.astype(np.uint8)
        else:
            assert img_idx in self.color_index
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target



class CIFAR_SuperClass(datasets.CIFAR10):
    def __init__(
        self,
        root,
        transform,
        target_transform,
        train,
        download,
        front_ratio, # portion of samples from first half of classes [0-4]
        back_ratio, # portion of samples from first half of classes [5-9]
        split,
        date
    ):
        super().__init__(root, train, transform, target_transform, download)

        self.front_number = int(front_ratio * 5000)
        self.back_number = int(back_ratio * 5000)

        self.full_by_class = {
            0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []
        }
        for index, label in enumerate(self.targets):
            self.full_by_class[label].append(index)
        for label in range(10):
            assert len(self.full_by_class[label]) == 5000, "there should be 5000 images per class in full cifar dataset."

        if split:
            self.part_by_class = {
                0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []
            }
            for label in range(0, 5):
                tmp_arr = np.array(self.full_by_class[label], dtype=int)
                self.part_by_class[label] = np.random.choice(tmp_arr, self.front_number, replace=False)
            for label in range(5, 10):
                tmp_arr = np.array(self.full_by_class[label], dtype=int)
                self.part_by_class[label] = np.random.choice(tmp_arr, self.back_number, replace=False)
            
            file_path = os.path.join(root, "class_split", date, "front{}_back{}_index.pkl".format(front_ratio, back_ratio))
            with open(file_path, "wb") as f:
                pickle.dump(self.part_by_class, f)
        
        else:
            file_path = os.path.join(root, "class_split", date, "front{}_back{}_index.pkl".format(front_ratio, back_ratio))
            with open(file_path, "rb") as f:
                file_load = pickle.load(f)
            print("Loading front/back index by class from file path {}".format(file_path))

            self.part_by_class = file_load # this should be a dict

            self.match_list = []
            for label in range(10):
                self.match_list.extend(self.part_by_class[label].tolist())
            random.shuffle(self.match_list)
            assert len(self.match_list) == 25000, "there should be 25k items in splitted subset"

    
    def __len__(self):
        # 25k training samples by default
        return int(super().__len__() / 2)
    
    def __getitem__(self, index):
        match_index = self.match_list[index]
        image, target = super().__getitem__(match_index)
        target = SUPER_CLASSES_INVERSE[target]
        return image, target
        
    

if __name__ == "__main__":
    # generate_index_cifar10()
    # check_cifar10_index()
    # generate_fixgroup_cifar10()
    # check_fixgroup_cifar10()
    # split_by_class(date="2023-04-20")
    # split_by_class(date="2023-04-21")
    check_by_class(date="2023-04-20")
    check_by_class(date="2023-04-21")
    