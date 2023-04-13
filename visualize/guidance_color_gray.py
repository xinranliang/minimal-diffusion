import os
import cv2
from PIL import Image
import copy
import math
import argparse
import numpy as np
from tqdm import tqdm


def get_args_guidance():
    parser = argparse.ArgumentParser("Qualitative inspection of underrepresented groups and overrepresented groups samples")

    parser.add_argument('--dataset', type=str, help="which dataset we use for training and evaluation")
    parser.add_argument('--train-color', type=float, help="manually set portion of color images in training distribution")
    parser.add_argument('--train-gray', type=float, help="manually set portion of gray images in training distribution")

    parser.add_argument('--num-samples', type=int, default=50000, help="number of samples available in total")
    parser.add_argument('--num-visualize', type=int, default=100, help="number of images to visualize, should be square number n^2")

    parser.add_argument('--date', type=str, help="experiment date for logging purpose")
    parser.add_argument('--diffusion-config', type=str, help="diffusion model training configuration")
    parser.add_argument('--sample-config', type=str, help="diffusion model sampling configuration, without .npz ending")
    parser.add_argument('--class-cond', action="store_true", default=True, help="whether diffusion model is trained to be class conditional")
    parser.add_argument('--num-classes', type=int, help="number of classes in dataset")

    args = parser.parse_args()

    return args 


def main_guidance():
    args = get_args_guidance()
    w_lst = [0.0, 0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0, 8.0]

    for class_label in range(args.num_classes):
        for w in w_lst:
            args.sample_file = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs", 
                                args.date, args.dataset, "color{}_gray{}".format(args.train_color, args.train_gray), 
                                args.diffusion_config, "samples_ema",
                                "guidance", "class_{}".format(class_label),
                                "{}_guidance{}.npz".format(args.sample_config, w))
            
            save_file = np.load(args.sample_file, allow_pickle=True)
            # samples shape: num_samples x height x width x n_channels
            samples, labels = save_file['arr_0'], save_file['arr_1']
            index = np.random.choice(args.num_samples, args.num_visualize, replace=False)
            samples = samples[index] # shape = num_samples x height x width x n_channel
            samples = np.split(samples, np.sqrt(args.num_visualize).astype(int), axis=0)
            viz_arr = np.concatenate(np.concatenate(samples, axis=1), axis=1)

            file_path = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs", 
                                args.date, args.dataset, "color{}_gray{}".format(args.train_color, args.train_gray), 
                                args.diffusion_config, "samples_ema", "guidance", "class_{}".format(class_label),
                                "{}_guidance{}_viz{}.png".format(args.sample_config, w, args.num_visualize))
            
            cv2.imwrite(
                file_path,
                viz_arr[:, :, ::-1]
            )


if __name__ == "__main__":
    main_guidance()