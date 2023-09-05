import os
import cv2
from PIL import Image
import copy
import math
import argparse
import numpy as np
from tqdm import tqdm

GENDER = {"Female": 0, "Male": 1}
RACE = (
    0, 1, 2, 3, 4, 5
)
gender_ratio = (
    0.1, 0.3, 0.5, 0.7, 0.9
)

num_visualize_per_class = 20

def visualize_fairface_gender(date, random_sample=True):
    ws = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    for female_ratio in gender_ratio:
        male_ratio = round(1.0 - female_ratio, 1)

        for w in ws:
            # sample npz file
            sample_file = os.path.join(
                "/n/fs/xl-diffbia/projects/minimal-diffusion/logs", date, "fairface", f"female{female_ratio}_male{male_ratio}",
                "UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_128_dropprob_0.1/samples_ema",
                f"fairface_gendersubset_f{female_ratio}_m{male_ratio}_ema_num50000_guidance{w}.npz"
            )
            load_file = np.load(sample_file, allow_pickle=True)
            # samples shape: num_samples x height x width x n_channels
            samples, labels = load_file['arr_0'], load_file['arr_1']
            viz_arr = np.zeros((len(RACE), num_visualize_per_class, samples.shape[1], samples.shape[2], samples.shape[3]), dtype=np.uint8)

            for idx, cls_idx in enumerate(RACE):
                sample_index = np.argwhere(labels == cls_idx).reshape(-1)
                if random_sample:
                    sample_index = np.random.choice(sample_index, size=num_visualize_per_class, replace=False)
                else:
                    sample_index = sample_index[:num_visualize_per_class]

                for viz_idx in range(num_visualize_per_class):
                    new_image, new_label = samples[sample_index[viz_idx]], labels[sample_index[viz_idx]]
                    np.copyto(dst=viz_arr[idx, viz_idx], src=new_image)
            
            viz_arr = np.concatenate(np.concatenate(viz_arr, axis=1), axis=1)

            save_path = os.path.join(
                "/n/fs/xl-diffbia/projects/minimal-diffusion/logs", date, "fairface", f"female{female_ratio}_male{male_ratio}",
                "UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_128_dropprob_0.1/visualize"
            )
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(
                os.path.join(save_path, f"female{female_ratio}_male{male_ratio}_guidance{w}.png"),
                viz_arr[:, :, ::-1]
            )


if __name__ == "__main__":
    visualize_fairface_gender(date="2023-07-25")
    visualize_fairface_gender(date="2023-07-26")