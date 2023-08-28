import numpy as np 
import os 
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

num_bins = 50
guidance_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0]

def plot_cifar_imgnet_hist(real_pred_probs, syn_pred_probs, save_folder):
    assert len(guidance_values) == len(syn_pred_probs), "Guidance values do not match"

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        # plt.hist(real_pred_probs, bins=num_bins, histtype="step", label="real samples")
        for idx in range(len(guidance_values)):
            plt.hist(syn_pred_probs[idx], bins=num_bins, histtype="step", label=f"$w = {guidance_values[idx]}$")
        plt.xticks(np.arange(0, 1.01, 0.1), np.arange(0, 1.01, 0.1))
        plt.xlabel("Probability value")
        plt.ylabel("Frequency")
        plt.title("Empirical histogram of predicted probability as from CIFAR domain")
        plt.legend(loc="upper center")
    
    plt.savefig(os.path.join(save_folder, "pred_prob_hist.png"), dpi=300, bbox_inches="tight")

def plot_cifar_imgnet():
    colors = ["blue", "orange", "green", "purple", "brown"]
    domain_split = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float64)
    ws = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    counts = [
        np.array([0.119, 0.122, 0.128, 0.132, 0.136, 0.135, 0.135, 0.123, 0.109], dtype=float),
        np.array([0.372, 0.391, 0.413, 0.428, 0.439, 0.443, 0.454, 0.431, 0.391], dtype=float),
        np.array([0.572, 0.6, 0.625, 0.64, 0.651, 0.66, 0.665, 0.633, 0.585], dtype=float),
        np.array([0.756, 0.777, 0.794, 0.809, 0.819, 0.823, 0.821, 0.786, 0.732], dtype=float),
        np.array([0.931, 0.938, 0.941, 0.945, 0.943, 0.943, 0.923, 0.884, 0.823], dtype=float)
    ]

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        for idx in range(len(domain_split)):
            plt.plot(ws, counts[idx], linestyle="-", linewidth=1, color=colors[idx])
            plt.axhline(domain_split[idx], xmin=ws[0], xmax=ws[-1], linestyle="--", color=colors[idx])
        plt.xticks(ws, ws)
        plt.ylim(0, 1.01)
        plt.yticks(np.arange(0, 1.01, step=0.1))
        plt.xlabel("Scale of classifier-free guidance ($w$)")
        plt.ylabel("Value")
        plt.title("Sampling distribution of classified as CIFAR domain w.r.t Guidance")
        plt.legend()
    
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-31/cifar-imagenet/figures", "guidance_count.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-31/cifar-imagenet/figures", "guidance_count.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    os.makedirs("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-31/cifar-imagenet/figures", exist_ok=True)
    plot_cifar_imgnet()