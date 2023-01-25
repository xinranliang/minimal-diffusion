import os
import cv2
import copy
import math
import argparse
import numpy as np
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser("Qualitative inspection of underrepresented groups and overrepresented groups samples")

    parser.add_argument('--dataset', type=str, help="which dataset we use for training and evaluation")
    parser.add_argument('--train-color', type=float, help="manually set portion of color images in training distribution")
    parser.add_argument('--train-gray', type=float, help="manually set portion of gray images in training distribution")

    parser.add_argument('--sample-color', action="store_true", help="only inspect color samples")
    parser.add_argument('--sample-gray', action="store_true", help="only inspect gray samples")
    parser.add_argument('--num-samples', type=int, help="number of samples in total")

    parser.add_argument('--num-visualize', type=int, default=100, help="number of images to visualize, should be square number n^2")

    parser.add_argument('--date', type=str, help="experiment date for logging purpose")
    parser.add_argument('--diffusion-config', type=str, default="UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512", help="diffusion model configuration, currently set to be default parameters")

    args = parser.parse_args()

    if args.sample_color:
        args.sample_str = "color"
    elif args.sample_gray:
        args.sample_str = "gray"
    else:
        args.sample_str = "none"
    
    if args.sample_str == "none":
        args.sample_file = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs", 
                        args.date, args.dataset, "color{}_gray{}".format(args.train_color, args.train_gray), 
                        args.diffusion_config, "samples", 
                        "{}_color{}_gray{}_epoch_{}_num{}.npz".format(args.dataset, args.train_color, args.train_gray, 950, args.num_samples))
    else:
        args.sample_file = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs", 
                            args.date, args.dataset, "color{}_gray{}".format(args.train_color, args.train_gray), 
                            args.diffusion_config, "samples", 
                            "{}_color{}_gray{}_epoch_{}_num{}_{}.npz".format(args.dataset, args.train_color, args.train_gray, 950, args.num_samples, args.sample_str))

    return args 


def main():
    args = get_args()

    save_file = np.load(args.sample_file, allow_pickle=True)
    index = np.random.choice(args.num_samples, args.num_visualize, replace=False)
    samples = save_file['arr_0'][index] # shape = num_samples x height x width x n_channel
    samples = np.split(samples, np.sqrt(args.num_visualize).astype(int), axis=0)

    if args.sample_str == "none":
        file_path = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs", 
                            args.date, args.dataset, "color{}_gray{}".format(args.train_color, args.train_gray), 
                            args.diffusion_config, "figures",
                            "{}_color{}_gray{}_epoch_{}.png".format(args.dataset, args.train_color, args.train_gray, 950))
    else:
        file_path = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs", 
                                args.date, args.dataset, "color{}_gray{}".format(args.train_color, args.train_gray), 
                                args.diffusion_config, "figures",
                                "{}_color{}_gray{}_epoch_{}_{}.png".format(args.dataset, args.train_color, args.train_gray, 950, args.sample_str))

    cv2.imwrite(
        file_path,
        np.concatenate(np.concatenate(samples, axis=1), axis=1)[:, :, ::-1]
    )


if __name__ == "__main__":
    main()