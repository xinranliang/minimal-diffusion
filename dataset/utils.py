import numpy as np

import cv2
import os
import csv
from collections import OrderedDict
from time import time

import torch
from torch.utils.tensorboard import SummaryWriter


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