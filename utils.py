import numpy as np
import cv2
import os
import csv
from collections import OrderedDict
from time import time

import torch
from torch.utils.tensorboard import SummaryWriter


def rgb_to_grayscale(image):
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


def prob_mask_shapelike(shape, keep_prob, device=torch.device("cuda")):
    if keep_prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif keep_prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < keep_prob


class logger(object):
    def __init__(self, max_steps, writer_mode, log_dir, ema_w):
        self.max_steps = max_steps
        self.start_time = time()
        self.ema_loss = None
        self.ema_w = ema_w

        if torch.cuda.current_device() == 0:
            if "csv" in writer_mode:
                self.csv_file = open(os.path.join(log_dir, "log.csv"), 'w')
                self.csv_writer = None
            if "tb" in writer_mode:
                self.tb_writer = SummaryWriter(log_dir)
            if "txt" in writer_mode:
                self.txt_file = open(os.path.join(log_dir, "log.txt"), 'w')

    def log(self, value, step):
        if self.ema_loss is None:
            self.ema_loss = value
        else:
            self.ema_loss = self.ema_w * self.ema_loss + (1 - self.ema_w) * value
        
        data_dict = {"loss": value, "ema_loss": self.ema_loss, "step": step}
        
        # log to tb
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("loss", data_dict["loss"], step)
            self.tb_writer.add_scalar("ema_loss", self.ema_loss, step)
        
        # log to csv and txt
        if step % 20 == 0:

            if self.csv_file is not None:
                if self.csv_writer is None:
                    self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=sorted(data_dict.keys()), restval=0.0)
                    self.csv_writer.writeheader()
                self.csv_writer.writerow(data_dict)
                self.csv_file.flush()
            
            if self.txt_file is not None:
                self.txt_file.write("Step: {} loss: {} ema_loss: {} \n".format(step, data_dict["loss"], data_dict["ema_loss"]))
                self.txt_file.flush()

        # dump to console
        if step % 100 == 0:
            print(
                f"Steps: {step}/{self.max_steps} \t loss: {value} \t loss (ema): {self.ema_loss:.3f} "
                + f"\t Time elapsed: {(time() - self.start_time)/3600:.3f} hr"
            )


def remove_module(d):
    return OrderedDict({(k[len("module.") :], v) for (k, v) in d.items()})


def fix_legacy_dict(d):
    keys = list(d.keys())
    if "model" in keys:
        d = d["model"]
    if "state_dict" in keys:
        d = d["state_dict"]
    keys = list(d.keys())
    # remove multi-gpu module.
    if "module." in keys[1]:
        d = remove_module(d)
    return d

unsqueeze3x = lambda x: x[..., None, None, None]