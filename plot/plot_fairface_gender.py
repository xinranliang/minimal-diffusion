import numpy as np 
import os 
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

num_bins = 50
guidance_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

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


def plot_fairface(correct, run):
    colors = ["blue", "orange", "green", "purple", "brown"]
    domain_split = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float64)
    ws = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)

    if run == 1:
        counts = [
            np.array([0.143, 0.123, 0.113, 0.104, 0.099, 0.096, 0.092, 0.091, 0.087, 0.087, 0.087], dtype=float),
            np.array([0.333, 0.324, 0.319, 0.309, 0.308, 0.305, 0.305, 0.299, 0.295, 0.291, 0.293], dtype=float),
            np.array([0.513, 0.504, 0.494, 0.468, 0.448, 0.422, 0.403, 0.384, 0.364, 0.354, 0.349], dtype=float),
            np.array([0.692, 0.695, 0.692, 0.666, 0.650, 0.625, 0.598, 0.583, 0.565, 0.553, 0.546], dtype=float),
            np.array([0.867, 0.880, 0.884, 0.878, 0.870, 0.859, 0.842, 0.832, 0.817, 0.807, 0.798], dtype=float),
        ]
    elif run == 2:
        counts = [
            np.array([0.147, 0.140, 0.135, 0.132, 0.136, 0.132, 0.132, 0.130, 0.124, 0.124, 0.122], dtype=float),
            np.array([0.329, 0.318, 0.308, 0.292, 0.288, 0.278, 0.271, 0.264, 0.259, 0.256, 0.258], dtype=float),
            np.array([0.518, 0.524, 0.531, 0.523, 0.515, 0.502, 0.490, 0.477, 0.471, 0.464, 0.452], dtype=float),
            np.array([0.690, 0.705, 0.706, 0.707, 0.701, 0.689, 0.678, 0.668, 0.655, 0.652, 0.646], dtype=float),
            np.array([0.870, 0.888, 0.900, 0.902, 0.900, 0.892, 0.887, 0.875, 0.864, 0.857, 0.849], dtype=float),
        ]

    if correct:
        coeff_lower = [
            np.array([0.364, 0.071 + (0.364 - 0.071) / 5 * 1, 0.071 + (0.364 - 0.071) / 5 * 2, 0.071 + (0.364 - 0.071) / 5 * 3, 0.071 + (0.364 - 0.071) / 5 * 4, 0.071, 0.071 + (0.727 - 0.071) / 5 * 1, 0.071 + (0.727 - 0.071) / 5 * 2, 0.071 + (0.727 - 0.071) / 5 * 3, 0.071 + (0.727 - 0.071) / 5 * 4, 0.727], dtype=float),
            np.array([0.474, 0.291 + (0.474 - 0.291) / 5 * 1, 0.291 + (0.474 - 0.291) / 5 * 2, 0.291 + (0.474 - 0.291) / 5 * 3, 0.291 + (0.474 - 0.291) / 5 * 4, 0.291, 0.291 + (0.769 - 0.291) / 5 * 1, 0.291 + (0.769 - 0.291) / 5 * 2, 0.291 + (0.769 - 0.291) / 5 * 3, 0.291 + (0.769 - 0.291) / 5 * 4, 0.769], dtype=float),
            np.array([0.585, 0.511 + (0.585 - 0.511) / 5 * 1, 0.511 + (0.585 - 0.511) / 5 * 2, 0.511 + (0.585 - 0.511) / 5 * 3, 0.511 + (0.585 - 0.511) / 5 * 4, 0.511, 0.511 + (0.811 - 0.511) / 5 * 1, 0.511 + (0.811 - 0.511) / 5 * 2, 0.511 + (0.811 - 0.511) / 5 * 3, 0.511 + (0.811 - 0.511) / 5 * 4, 0.811], dtype=float),
            np.array([0.640, 0.640 + (0.687 - 0.640) / 5 * 1, 0.640 + (0.687 - 0.640) / 5 * 2, 0.640 + (0.687 - 0.640) / 5 * 3, 0.640 + (0.687 - 0.640) / 5 * 4, 0.687, 0.687 + (0.912 - 0.687) / 5 * 1, 0.687 + (0.912 - 0.687) / 5 * 2, 0.687 + (0.912 - 0.687) / 5 * 3, 0.687 + (0.912 - 0.687) / 5 * 4, 0.912], dtype=float),
            np.array([0.694, 0.694 + (0.864 - 0.694) / 5 * 1, 0.694 + (0.864 - 0.694) / 5 * 2, 0.694 + (0.864 - 0.694) / 5 * 3, 0.694 + (0.864 - 0.694) / 5 * 4, 0.864, 0.864 + (1.013 - 0.864) / 5 * 1, 0.864 + (1.013 - 0.864) / 5 * 2, 0.864 + (1.013 - 0.864) / 5 * 3, 0.864 + (1.013 - 0.864) / 5 * 4, 1.013], dtype=float),
        ]
        coeff_upper = [
            np.array([0.694, 0.929 + (1.273 - 0.929) / 5 * 1, 0.929 + (1.273 - 0.929) / 5 * 2, 0.929 + (1.273 - 0.929) / 5 * 3, 0.929 + (1.273 - 0.929) / 5 * 4, 0.929, 0.929 + (1.455 - 0.929) / 5 * 1, 0.929 + (1.455 - 0.929) / 5 * 2, 0.929 + (1.455 - 0.929) / 5 * 3, 0.929 + (1.455 - 0.929) / 5 * 4, 1.455], dtype=float),
            np.array([1.136, 0.898 + (1.136 - 0.898) / 5 * 1, 0.898 + (1.136 - 0.898) / 5 * 2, 0.898 + (1.136 - 0.898) / 5 * 3, 0.898 + (1.136 - 0.898) / 5 * 4, 0.898, 0.898 + (1.241 - 0.898) / 5 * 1, 0.898 + (1.241 - 0.898) / 5 * 2, 0.898 + (1.241 - 0.898) / 5 * 3, 0.898 + (1.241 - 0.898) / 5 * 4, 1.241], dtype=float),
            np.array([1.000, 0.867 + (1.000 - 0.867) / 5 * 1, 0.867 + (1.000 - 0.867) / 5 * 2, 0.867 + (1.000 - 0.867) / 5 * 3, 0.867 + (1.000 - 0.867) / 5 * 4, 0.867, 0.867 + (1.027 - 0.867) / 5 * 1, 0.867 + (1.027 - 0.867) / 5 * 2, 0.867 + (1.027 - 0.867) / 5 * 3, 0.867 + (1.027 - 0.867) / 5 * 4, 1.027], dtype=float),
            np.array([1.024, 0.967 + (1.024 - 0.967) / 5 * 1, 0.967 + (1.024 - 0.967) / 5 * 2, 0.967 + (1.024 - 0.967) / 5 * 3, 0.967 + (1.024 - 0.967) / 5 * 4, 0.967, 0.967 + (1.083 - 0.967) / 5 * 1, 0.967 + (1.083 - 0.967) / 5 * 2, 0.967 + (1.083 - 0.967) / 5 * 3, 0.967 + (1.083 - 0.967) / 5 * 4, 1.083], dtype=float),
            np.array([1.047, 1.047 + (1.068 - 1.047) / 5 * 1, 1.047 + (1.068 - 1.047) / 5 * 2, 1.047 + (1.068 - 1.047) / 5 * 3, 1.047 + (1.068 - 1.047) / 5 * 4, 1.068, 1.068 + (1.139 - 1.068) / 5 * 1, 1.068 + (1.139 - 1.068) / 5 * 2, 1.068 + (1.139 - 1.068) / 5 * 3, 1.068 + (1.139 - 1.068) / 5 * 4, 1.139], dtype=float),
        ]

        counts_lower, counts_upper = [], []
        for idx in range(len(domain_split)):
            counts_lower.append(counts[idx] * coeff_lower[idx])
            counts_upper.append(counts[idx] * coeff_upper[idx])

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        for idx in range(len(domain_split)):
            plt.plot(ws, counts_lower[idx], linestyle="-", color=colors[idx])
            plt.plot(ws, counts_upper[idx], linestyle="-", color=colors[idx])
            plt.fill_between(ws, counts_upper[idx], counts_lower[idx], alpha=0.2, color=colors[idx])
            plt.axhline(domain_split[idx], xmin=ws[0], xmax=ws[-1], linestyle="--", color=colors[idx])
        plt.xticks(ws, ws)
        plt.ylim(0, 1.01)
        plt.yticks(np.arange(0, 1.01, step=0.1))
        plt.xlabel("Scale of classifier-free guidance ($w$)")
        plt.ylabel("Portion classified as Female samples")
        plt.title("Sampling distribution over 50k of classified as Female w.r.t Guidance on FairFace")
    
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-25/fairface/figures", f"guidance_count_gender_{run}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-25/fairface/figures", f"guidance_count_gender_{run}.pdf"), dpi=300, bbox_inches="tight")
    plt.close()



def plot_check_repr():
    ws = np.array([0.0, 5.0, 10.0], dtype=np.float64)
    colors = ["blue", "orange", "green", "purple", "brown"]
    line_labels = ["female/male = 0.1/0.9", "female/male = 0.5/0.5", "female/male = 0.9/0.1"]
    true_train = [
        np.array([0.1, 0.1, 0.1], dtype=np.float64), # 0.1/0.9
        np.array([0.5, 0.5, 0.5], dtype=np.float64), # 0.5/0.5
        np.array([0.9, 0.9, 0.9], dtype=np.float64) # 0.9/0.1
    ]
    true_repr = [
        np.array([18, 6, 12], dtype=np.float64) / 100, # 0.1/0.9
        np.array([42, 31, 34], dtype=np.float64) / 100, # 0.5/0.5
        np.array([74, 85, 85], dtype=np.float64) / 100 # 0.9/0.1
    ]
    pred_repr = [
        np.array([22, 14, 11], dtype=np.float64) / 100, # 0.1/0.9
        np.array([53, 45, 37], dtype=np.float64) / 100, # 0.5/0.5
        np.array([85, 88, 79], dtype=np.float64) / 100 # 0.9/0.1
    ]
    unsure_repr = [
        np.array([10, 7, 4], dtype=np.float64) / 100, # 0.1/0.9
        np.array([11, 8, 4], dtype=np.float64) / 100, # 0.5/0.5
        np.array([15, 9, 5], dtype=np.float64) / 100 # 0.9/0.1
    ]

    plt.figure(figsize=(8, 6))
    with plt.style.context('ggplot'):
        for idx in range(len(line_labels)):
            plt.plot(ws, pred_repr[idx], linestyle="-", color=colors[2 * idx])
            plt.plot(ws, true_repr[idx] + unsure_repr[idx], linestyle="--", color=colors[2 * idx])
            plt.plot(ws, true_repr[idx] - unsure_repr[idx], linestyle="--", color=colors[2 * idx])
            plt.fill_between(ws, true_repr[idx] + unsure_repr[idx], true_repr[idx] - unsure_repr[idx], alpha=0.2, color=colors[2 * idx])
        plt.xticks(ws, ws)
        plt.ylim(0, 1.01)
        plt.yticks(np.arange(0, 1.01, step=0.05))
        plt.xlabel("Scale of classifier-free guidance ($w$)")
        plt.ylabel("Empirical predicted v.s. annotated Female representation")
        plt.title("Automatic gender classifier on FairFace w.r.t Guidance")
    
    plt.savefig("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-25/fairface/figures/acc_repr_level.png", dpi=300, bbox_inches="tight")
    plt.savefig("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-25/fairface/figures/acc_repr_level.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-25/fairface/figures", exist_ok=True)
    # report results
    plot_fairface(correct=True, run=1)
    plot_fairface(correct=True, run=2)
    # check reported results
    # plot_check_repr()