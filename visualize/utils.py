import numpy as np
import cv2
import os

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