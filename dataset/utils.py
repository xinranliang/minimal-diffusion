import numpy as np

import cv2
from PIL import Image
import os
import csv
from collections import OrderedDict
from time import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def rgb_to_gray(image):
    """
    Function to convert RGB channel images to grayscale images, still preserve shape/dimension

    Input: height x width x 3
    Output: height x width x 3

    """

    gray_image = np.zeros(image.shape)
    R = np.array(image[:, :, 0])
    G = np.array(image[:, :, 1])
    B = np.array(image[:, :, 2])

    avg_channel = R * 0.299 + G * 0.587 + B * 0.114

    for i in range(3):
        gray_image[:,:,i] = avg_channel
           
    return gray_image


class logger(object):
    def __init__(self, max_steps, writer_mode, log_dir):
        self.max_steps = max_steps
        self.start_time = time()

        if torch.cuda.current_device() == 0:
            if "csv" in writer_mode:
                self.csv_file = open(os.path.join(log_dir, "log.csv"), 'w')
                self.csv_writer = None
            else:
                self.csv_file = None
            if "tb" in writer_mode:
                self.tb_writer = SummaryWriter(log_dir)
            else:
                self.tb_writer = None 
            if "txt" in writer_mode:
                self.txt_file = open(os.path.join(log_dir, "log.txt"), 'w')
            else:
                self.txt_file = None

    def log(self, data_dict, step):
        # log to tb
        if self.tb_writer is not None:
            for key, value in data_dict.items():
                self.tb_writer.add_scalar(key, value, step)
        data_dict.update({"step": step})
        
        # log to csv and txt
        if step % 20 == 0:

            if self.csv_file is not None:
                if self.csv_writer is None:
                    self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=sorted(data_dict.keys()), restval=0.0)
                    self.csv_writer.writeheader()
                self.csv_writer.writerow(data_dict)
                self.csv_file.flush()
            
            if self.txt_file is not None:
                self.txt_file.write("Step: {} loss: {} accuracy: {} \n".format(step, data_dict["pred_loss"], data_dict["pred_acc"]))
                self.txt_file.flush()

        # dump to console
        if step % 100 == 0:
            print(
                "Steps: %d/%d \t loss: %.3f \t accuracy: %.3f " % (step, self.max_steps, data_dict["pred_loss"], data_dict["pred_acc"])
                + f"\t Time elapsed: {(time() - self.start_time)/3600:.3f} hr"
            )


class ArraytoImage(Dataset):
    def __init__(
        self,
        paths,
        transform,
        target_transform
    ):
        super().__init__()
        
        # load images from numpy array
        self.images, self.labels, self.targets = [], [], []
        for path in paths:
            if path.endswith(".npy"):
                new_images = np.load(path)
            elif path.endswith("npz"):
                file_load = np.load(path, allow_pickle=True)
                new_images, new_labels = file_load["arr_0"], file_load["arr_1"]
            else:
                raise ValueError(f"Unrecognized file type: {path}")
            self.images.append(new_images)
            self.labels.append(new_labels)
            if "imagenet0" in path:
                # generative model trained only on cifar real samples
                new_targets = np.zeros_like(new_labels, dtype=int)
            elif "cifar0" in path:
                # generative model trained only on imagenet real samples
                new_targets = np.ones_like(new_labels, dtype=int)
            else:
                raise NotImplementedError
            self.targets.append(new_targets)
        self.images = np.concatenate(self.images, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)
        
        # assert in valid form
        if self.images.min() >= 0 and self.images.max() <= 1:
            self.images = (self.images * 255).astype("uint8")
        elif self.images.min() >= -1 and self.images.max() <= 1:
            self.images = (127.5 * (self.images + 1)).astype("uint8")
        else:
            assert self.images.min() >= 0 and self.images.max() <= 255
        
        assert len(self.images.shape) == 4, "Images must be a batch"
        assert self.images.shape[0] == self.labels.shape[0], "Number of images and labels must match"
        self.num_items = self.images.shape[0]

        # transformation
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return self.num_items
    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index] # height x width x 3 (RGB order)
        image = Image.fromarray(image, mode="RGB")

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        target = self.targets[index]
        
        return image, label, target # image, class label, cifar/imagenet domain


class ArrayToImageLabel(Dataset):
    def __init__(
        self,
        samples,
        labels,
        mode, # RGB or L-gray
        transform,
        target_transform
    ):
        super().__init__()

        # convert from numpy array
        self.samples = samples
        if labels is not None:
            self.labels = labels
        else:
            self.labels = None
        # assert in valid form
        if self.samples.min() >= 0 and self.samples.max() <= 1:
            self.samples = (self.samples * 255).astype("uint8")
        elif self.samples.min() >= -1 and self.samples.max() <= 1:
            self.samples = (127.5 * (self.samples + 1)).astype("uint8")
        else:
            assert self.samples.min() >= 0 and self.samples.max() <= 255
        
        assert len(self.samples.shape) == 4, "Images must be a batch"
        assert self.samples.shape[0] == self.labels.shape[0], "Number of images and labels must match"
        self.num_items = self.samples.shape[0]

        # transformation
        self.transform = transform
        self.target_transform = target_transform

        # 3 channel or 1 channel
        self.mode = mode

    def __len__(self):
        return self.num_items
    
    def __getitem__(self, index):
        image = self.samples[index] # height x width x num_channels
        if self.labels is not None:
            label = self.labels[index]
        image = Image.fromarray(np.squeeze(image), mode=self.mode)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        if self.labels is not None:
            return image, label # image, class-condition label
        else:
            return image