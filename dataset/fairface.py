import argparse
import os
import pickle
import numpy as np
import random
from collections import defaultdict
import csv
from PIL import Image

import torch
import torch.nn as nn 
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

root_path = "/n/fs/visualai-scr/Data/FairFace"

GENDER = {"Female": 0, "Male": 1}
RACE = {
    "White": 0, 
    "Black": 1, 
    "Indian": 2, 
    "East Asian": 3, 
    "Southeast Asian": 4, 
    # "Middle Eastern": 5, this race does not have balanced gender distribution
    "Latino_Hispanic": 5
}
RACE_FULL = {
    "White": 0, 
    "Black": 1, 
    "Indian": 2, 
    "East Asian": 3, 
    "Southeast Asian": 4, 
    "Middle Eastern": 5,
    "Latino_Hispanic": 6
}

BASE_NUM = 5000
# resolution = 224 x 224

def split_gender(date):
    trainsform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        )
    # full base
    myff = FairFace_Gender("train", trainsform_train, 1.0, 1.0, init=True, date=date)
    for f_ratio, m_ratio in zip([0.9, 0.7, 0.5, 0.3, 0.1], [0.1, 0.3, 0.5, 0.7, 0.9]):
        myff = FairFace_Gender("train", trainsform_train, f_ratio, m_ratio, init=True, date=date)
    for f_ratio, m_ratio in zip([0.9, 0.7, 0.5, 0.3, 0.1], [0.1, 0.3, 0.5, 0.7, 0.9]):
        myff = FairFace_Gender("train", trainsform_train, f_ratio, m_ratio, init=False, date=date)
        # check duplicate
        assert len(myff.part_index) == len(set(myff.part_index)), "there are duplicate items in index!"

class FairFace_Gender(Dataset):
    def __init__(self, split, transform, f_ratio, m_ratio, root=root_path, init=False, date=None):
        super().__init__()
        self.split = split
        self.transform = transform

        self.root_path = root
        # image_path, gender, race
        self.image_paths = []
        self.gender_labels = []
        self.race_labels = []

        # load csv
        if split == "train":
            if init:
                self.full_index = {}
                self.part_index = {}
                for race in RACE.values():
                    self.full_index[race] = {}
                    self.part_index[race] = {}
                    for gender in GENDER.values():
                        self.full_index[race][gender] = []

            with open(os.path.join(self.root_path, "fairface_label_train.csv"), "r") as f:
                csv_reader = csv.reader(f, delimiter=",")

                header = next(csv_reader)
                file_idx = header.index("file")
                gender_idx = header.index("gender")
                race_idx = header.index("race")

                curr_index = 0
                for row in csv_reader:
                    if row[race_idx] != "Middle Eastern":
                        self.image_paths.append(os.path.join(self.root_path, row[file_idx]))
                        self.gender_labels.append(GENDER[row[gender_idx]])
                        self.race_labels.append(RACE[row[race_idx]])
                        if init and f_ratio == 1.0 and m_ratio == 1.0:
                            self.full_index[RACE[row[race_idx]]][GENDER[row[gender_idx]]].append(curr_index)
                        curr_index += 1

        elif split == "val":
            with open(os.path.join(self.root_path, "fairface_label_val.csv"), "r") as f:
                csv_reader = csv.reader(f, delimiter=",")

                header = next(csv_reader)
                file_idx = header.index("file")
                gender_idx = header.index("gender")
                race_idx = header.index("race")

                for row in csv_reader:
                    if row[race_idx] != "Middle Eastern":
                        self.image_paths.append(os.path.join(self.root_path, row[file_idx]))
                        self.gender_labels.append(GENDER[row[gender_idx]])
                        self.race_labels.append(RACE[row[race_idx]])
        else:
            raise ValueError("invalid dataset split")
        
        assert len(self.image_paths) == len(self.gender_labels) and len(self.image_paths) == len(self.race_labels), "number of samples does not match!"

        # subsample and save
        if init and f_ratio == 1.0 and m_ratio == 1.0:
            for race in RACE.values():
                for gender in GENDER.values():
                    self.part_index[race][gender] = np.random.choice(np.array(self.full_index[race][gender], dtype=int), size=BASE_NUM, replace=False)
                    # print(self.part_index[race][gender].shape)
            file_path = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/fairface/index_split", date)
            os.makedirs(file_path, exist_ok=True)
            file_path = os.path.join(file_path, "index_f1.0_m1.0.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(self.part_index, f)
        
        elif init:
            file_path = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/fairface/index_split", date, "index_f1.0_m1.0.pkl")
            with open(file_path, "rb") as f:
                self.full_index = pickle.load(f)
            for race in RACE.values():
                self.part_index[race][0] = np.random.choice(np.array(self.full_index[race][0], dtype=int), size =int(BASE_NUM * f_ratio), replace=False)
                self.part_index[race][1] = np.random.choice(np.array(self.full_index[race][1], dtype=int), size = int(BASE_NUM * m_ratio), replace=False)
                # print("Shape of female samples: ", self.part_index[race][0].shape)
                # print("Shape of male samples: ", self.part_index[race][1].shape)
            file_path = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/fairface/index_split", date, f"index_f{f_ratio}_m{m_ratio}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(self.part_index, f)
        
        else:
            file_path = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/fairface/index_split", date, f"index_f{f_ratio}_m{m_ratio}.pkl")
            with open(file_path, "rb") as f:
                file_load = pickle.load(f)
            print(f"Loading dataset from path {file_path}.")
            self.part_index = []
            for race in RACE.values():
                for gender in GENDER.values():
                    self.part_index.extend(file_load[race][gender])
            self.part_index = sorted(self.part_index)

            assert len(self.part_index) == BASE_NUM * len(RACE) * len(GENDER) // 2
    
    def __len__(self):
        if self.split == "train":
            return BASE_NUM * len(RACE) * len(GENDER) // 2
        elif self.split == "val":
            return len(self.image_paths)
        else:
            raise ValueError("invalid dataset split")
    
    def __getitem__(self, index):
        # get true index
        raw_idx = self.part_index[index]
        # get image
        img_path = self.image_paths[raw_idx]
        with open(img_path, "rb") as f:
            image = Image.open(f).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # get label: race, gender
        return image, self.race_labels[raw_idx], self.gender_labels[raw_idx]


class FairFace_Base(Dataset):
    def __init__(self, split, transform, root=root_path):
        super().__init__()
        self.transform = transform

        self.root_path = root
        # image_path, gender, race
        self.image_paths = []
        self.gender_labels = []
        self.race_labels = []

        # load csv
        if split == "train":
            with open(os.path.join(self.root_path, "fairface_label_train.csv"), "r") as f:
                csv_reader = csv.reader(f, delimiter=",")

                header = next(csv_reader)
                file_idx = header.index("file")
                gender_idx = header.index("gender")
                race_idx = header.index("race")

                for row in csv_reader:
                    self.image_paths.append(os.path.join(self.root_path, row[file_idx]))
                    self.gender_labels.append(GENDER[row[gender_idx]])
                    self.race_labels.append(RACE_FULL[row[race_idx]])
        
        elif split == "val":
            with open(os.path.join(self.root_path, "fairface_label_val.csv"), "r") as f:
                csv_reader = csv.reader(f, delimiter=",")

                header = next(csv_reader)
                file_idx = header.index("file")
                gender_idx = header.index("gender")
                race_idx = header.index("race")

                for row in csv_reader:
                    self.image_paths.append(os.path.join(self.root_path, row[file_idx]))
                    self.gender_labels.append(GENDER[row[gender_idx]])
                    self.race_labels.append(RACE_FULL[row[race_idx]])
        
        else:
            raise ValueError("invalid dataset split")
        
        assert len(self.image_paths) == len(self.gender_labels) and len(self.image_paths) == len(self.race_labels), "number of samples does not match!"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        # get image
        img_path = self.image_paths[index]
        with open(img_path, "rb") as f:
            image = Image.open(f).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # get label: race, gender
        return image, self.race_labels[index], self.gender_labels[index]


class FairFace_Race(Dataset):
    def __init__(self, root, split, transform):
        super().__init__()
        self.transform = transform

        self.root_path = root
        # image_path, gender, race
        self.image_paths = []
        self.gender_labels = []
        self.race_labels = []

        # load csv
        if split == "train":
            with open(os.path.join(self.root_path, "fairface_label_train.csv"), "r") as f:
                csv_reader = csv.reader(f, delimiter=",")

                header = next(csv_reader)
                file_idx = header.index("file")
                gender_idx = header.index("gender")
                race_idx = header.index("race")

                for row in csv_reader:
                    if row[race_idx] == "White" or row[race_idx] == "Black":
                        self.image_paths.append(os.path.join(self.root_path, row[file_idx]))
                        self.gender_labels.append(GENDER[row[gender_idx]])
                        self.race_labels.append(RACE[row[race_idx]])

        elif split == "val":
            with open(os.path.join(self.root_path, "fairface_label_val.csv"), "r") as f:
                csv_reader = csv.reader(f, delimiter=",")

                header = next(csv_reader)
                file_idx = header.index("file")
                gender_idx = header.index("gender")
                race_idx = header.index("race")

                for row in csv_reader:
                    if row[race_idx] == "White" or row[race_idx] == "Black":
                        self.image_paths.append(os.path.join(self.root_path, row[file_idx]))
                        self.gender_labels.append(GENDER[row[gender_idx]])
                        self.race_labels.append(RACE[row[race_idx]])
        else:
            raise ValueError("invalid dataset split")
        
        assert len(self.image_paths) == len(self.gender_labels) and len(self.image_paths) == len(self.race_labels), "number of samples does not match!"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        # get image
        img_path = self.image_paths[index]
        with open(img_path, "rb") as f:
            image = Image.open(f).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # get label: race, gender
        return image, self.race_labels[index], self.gender_labels[index]


if __name__ == "__main__":
    # split_gender(date="2023-07-25")
    # split_gender(date="2023-07-26")
    trainsform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        )
    myff_full = FairFace_Base("train", trainsform_train)
    print(len(myff_full))