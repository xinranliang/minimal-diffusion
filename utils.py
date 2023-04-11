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


def arr_to_img(load_path, save_path, args, random_sample=True):
    # load_path could be string or numpy array
    if isinstance(load_path, str):
        save_file = np.load(load_path, allow_pickle=True)
    elif isinstance(load_path, dict):
        save_file = load_path

    # cond, need to get (num_visualize // num_classes) samples per class
    if args.class_cond:
        assert args.num_classes > 0

        # samples shape: num_samples x height x width x n_channels
        samples, labels = save_file['arr_0'], save_file['arr_1']
        # visualize shape: num_classes x num_samples x height x width x n_channels
        num_viz_per_cls = int(args.num_visualize / args.num_classes)
        viz_arr = np.zeros((args.num_classes, num_viz_per_cls, samples.shape[1], samples.shape[2], samples.shape[3]), dtype=np.uint8)
        for cls_idx in range(args.num_classes):
            sample_index = np.argwhere(labels == cls_idx).reshape(-1)
            if random_sample:
                sample_index = np.random.choice(sample_index, size=num_viz_per_cls, replace=False)
            else:
                sample_index = sample_index[:num_viz_per_cls]
            for viz_idx in range(num_viz_per_cls):
                new_image, new_label = samples[sample_index[viz_idx]], labels[sample_index[viz_idx]]
                np.copyto(dst=viz_arr[cls_idx, viz_idx], src=new_image)
        
        viz_arr = np.concatenate(np.concatenate(viz_arr, axis=1), axis=1)

    # uncond, random sample
    else:
        if random_sample:
            index = np.random.choice(args.num_samples, args.num_visualize, replace=False)
        else:
            index = range(args.num_samples[:args.num_visualize])
        # get all samples and select subset index for visualization
        samples = save_file['arr_0'][index] # shape = num_samples x height x width x n_channel
        samples = np.split(samples, np.sqrt(args.num_visualize).astype(int), axis=0)
        viz_arr = np.concatenate(np.concatenate(samples, axis=1), axis=1)
    
    cv2.imwrite(
        save_path,
        viz_arr[:, :, ::-1]
    )


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