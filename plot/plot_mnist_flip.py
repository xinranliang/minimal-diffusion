import numpy as np 
import os 
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def plot_mnist_flip():
    colors = ["blue", "orange", "green", "purple", "brown"]
    domain_split = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float64)
    ws = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    counts = [
        np.vstack([[0.077, 0.065, 0.056, 0.05, 0.05, 0.043, 0.036, 0.031, 0.029], [0.077, 0.064, 0.056, 0.049, 0.047, 0.041, 0.034, 0.028, 0.023]]),
        np.vstack([[0.283, 0.273, 0.266, 0.258, 0.257, 0.253, 0.24, 0.229, 0.225], [0.283, 0.27, 0.262, 0.252, 0.251, 0.245, 0.229, 0.214, 0.207]]),
        np.vstack([[0.5, 0.503, 0.502, 0.499, 0.501, 0.502, 0.508, 0.504, 0.503], [0.495, 0.499, 0.499, 0.5, 0.501, 0.503, 0.513, 0.512, 0.517]]),
        np.vstack([[0.714, 0.725, 0.74, 0.742, 0.747, 0.755, 0.779, 0.786, 0.791], [0.716, 0.727, 0.744, 0.746, 0.749, 0.759, 0.782, 0.788, 0.793]]),
        np.vstack([[0.911, 0.92, 0.93, 0.937, 0.937, 0.943, 0.952, 0.954, 0.959], [0.919, 0.929, 0.94, 0.943, 0.946, 0.952, 0.963, 0.965, 0.967]]),
    ]

    counts_mean = [
        np.mean(single_count, axis=0) for single_count in counts 
    ]
    conts_error = [
        np.absolute(single_count[1] - single_count[0]) / 2 for single_count in counts
    ]

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        for idx in range(len(domain_split)):
            plt.errorbar(ws, counts_mean[idx], yerr=conts_error[idx], linestyle="-", linewidth=1, color=colors[idx], capsize=3)
            plt.axhline(domain_split[idx], xmin=ws[0], xmax=ws[-1], linestyle="--", color=colors[idx])
        plt.xticks(ws, ws)
        plt.ylim(0, 1.01)
        plt.yticks(np.arange(0, 1.01, step=0.1))
        plt.xlabel("Scale of classifier-free guidance ($w$)")
        plt.ylabel("Value")
        plt.title("Sampling distribution w.r.t Guidance on MNIST dataset")
        # plt.legend()
    
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-04-08/mnist-subset/figures", "guidance_count_noflip.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-04-08/mnist-subset/figures", "guidance_count_noflip.pdf"), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    os.makedirs("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-04-08/mnist-subset/figures", exist_ok=True)
    plot_mnist_flip()