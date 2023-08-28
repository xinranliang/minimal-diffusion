import numpy as np 
import os 
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

num_bins = 50
guidance_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0]

def plot_fairface_hist(real_pred_probs, syn_pred_probs, save_folder):
    assert len(guidance_values) == len(syn_pred_probs), "Guidance values do not match"

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        plt.hist(real_pred_probs, bins=num_bins, histtype="step", label="real samples")
        for idx in range(len(guidance_values)):
            plt.hist(syn_pred_probs[idx], bins=num_bins, histtype="step", label=f"$w = {guidance_values[idx]}$")
        plt.xticks(np.arange(0, 1.01, 0.1), np.arange(0, 1.01, 0.1))
        plt.xlabel("Probability value")
        plt.ylabel("Frequency")
        plt.title("Empirical histogram of predicted probability as Female images")
        plt.legend(loc="upper center")
    
    plt.savefig(os.path.join(save_folder, "pred_prob_hist.png"), dpi=300, bbox_inches="tight")


def plot_fairface():
    colors = ["blue", "orange", "green", "purple", "brown"]
    domain_split = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float64)
    ws = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    counts = [
        np.vstack([[0.143, 0.137, 0.136, 0.127, 0.131, 0.123, 0.113, 0.104, 0.099], [0.147, 0.147, 0.142, 0.141, 0.14, 0.14, 0.135, 0.132, 0.136]]),
        np.vstack([[0.333, 0.331, 0.326, 0.325, 0.328, 0.324, 0.319, 0.309, 0.308], [0.329, 0.326, 0.324, 0.32, 0.322, 0.318, 0.308, 0.292, 0.288]]),
        np.vstack([[0.513, 0.51, 0.509, 0.508, 0.51, 0.504, 0.494, 0.468, 0.448], [0.518, 0.519, 0.524, 0.521, 0.532, 0.524, 0.531, 0.523, 0.515]]),
        np.vstack([[0.692, 0.692, 0.696, 0.693, 0.7, 0.695, 0.692, 0.666, 0.65], [0.69, 0.69, 0.696, 0.7, 0.702, 0.705, 0.706, 0.707, 0.701]]),
        np.vstack([[0.867, 0.869, 0.873, 0.874, 0.879, 0.88, 0.884, 0.878, 0.87], [0.87, 0.874, 0.878, 0.883, 0.883, 0.888, 0.9, 0.902, 0.9]]),
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
        plt.title("Sampling distribution of classified as Female w.r.t Guidance on FairFace dataset")
        plt.legend()
    
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-25/fairface/figures", "guidance_count_gender.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-25/fairface/figures", "guidance_count_gender.pdf"), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    os.makedirs("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-25/fairface/figures", exist_ok=True)
    plot_fairface()