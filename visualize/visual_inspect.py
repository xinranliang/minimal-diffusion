import os
import cv2
from PIL import Image
import copy
import math
import argparse
import numpy as np
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser("Qualitative inspection of underrepresented groups and overrepresented groups samples")

    parser.add_argument('--dataset', type=str, help="which dataset we use for training and evaluation")
    parser.add_argument('--train-color', help="manually set portion of color images in training distribution")
    parser.add_argument('--train-gray', help="manually set portion of gray images in training distribution")

    parser.add_argument('--num-samples', type=int, default=50000, help="number of samples available in total")
    parser.add_argument('--num-visualize', type=int, default=200, help="number of images to visualize, should be square number n^2")

    parser.add_argument('--date', type=str, help="experiment date for logging purpose")
    parser.add_argument('--diffusion-config', type=str, help="diffusion model training configuration")
    parser.add_argument('--sample-config', type=str, help="diffusion model sampling configuration, without .npz ending")
    parser.add_argument('--class-cond', action="store_true", help="whether diffusion model is trained to be class conditional")
    parser.add_argument('--num-classes', type=int, help="number of classes in dataset")

    args = parser.parse_args()
    
    args.sample_file = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs", 
                            args.date, args.dataset, "color{}_gray{}".format(args.train_color, args.train_gray), 
                            args.diffusion_config, "samples_ema", 
                            "{}.npz".format(args.sample_config))

    return args 


def main():
    args = get_args()

    save_file = np.load(args.sample_file, allow_pickle=True)

    # cond, need to get (num_visualize // num_classes) samples per class
    if args.class_cond:
        assert args.num_classes > 0

        # samples shape: num_samples x height x width x n_channels
        samples, labels = save_file['arr_0'], save_file['arr_1'][:args.num_samples]
        # visualize shape: num_classes x num_samples x height x width x n_channels
        num_viz_per_cls = int(args.num_visualize / args.num_classes)
        viz_arr = np.zeros((args.num_classes, num_viz_per_cls, samples.shape[1], samples.shape[2], samples.shape[3]), dtype=np.uint8)
        for cls_idx in range(args.num_classes):
            sample_index = np.argwhere(labels == cls_idx).reshape(-1)
            sample_index = np.random.choice(sample_index, size=num_viz_per_cls, replace=False)
            for viz_idx in range(num_viz_per_cls):
                new_image, new_label = samples[sample_index[viz_idx]], labels[sample_index[viz_idx]]
                np.copyto(dst=viz_arr[cls_idx, viz_idx], src=new_image)
        
        viz_arr = np.concatenate(np.concatenate(viz_arr, axis=1), axis=1)

    # uncond, random sample
    else:
        index = np.random.choice(args.num_samples, args.num_visualize, replace=False)
        # get all samples and select subset index for visualization
        samples = save_file['arr_0'][index] # shape = num_samples x height x width x n_channel
        samples = np.split(samples, np.sqrt(args.num_visualize).astype(int), axis=0)
        viz_arr = np.concatenate(np.concatenate(samples, axis=1), axis=1)

    file_path = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs", 
                            args.date, args.dataset, "color{}_gray{}".format(args.train_color, args.train_gray), 
                            args.diffusion_config, "samples_ema", "figures",
                            "{}_viz{}.png".format(args.sample_config, args.num_visualize))
    os.makedirs(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs", 
                            args.date, args.dataset, "color{}_gray{}".format(args.train_color, args.train_gray), 
                            args.diffusion_config, "samples_ema", "figures"), exist_ok=True)

    cv2.imwrite(
        file_path,
        viz_arr[:, :, ::-1]
    )


if __name__ == "__main__":
    main()