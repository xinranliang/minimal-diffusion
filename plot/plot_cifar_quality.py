import numpy as np 
import os 
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def plot_cifar_quality():
    num_cifar = 50000 / 1000
    xs = np.array([0, 25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000], dtype=int) // 1000

    # baseline model
    baseline_fid = np.vstack([[6.493, 6.376, 6.424, 6.442, 8.164, 6.629, 6.293, 6.401, 6.807], [6.118, 6.196, 6.231, 6.219, 6.96, 6.476, 6.337, 6.503, 6.629]])
    baseline_prec = np.vstack([[0.619, 0.617, 0.609, 0.609, 0.59, 0.601, 0.605, 0.605, 0.601], [0.627, 0.615, 0.612, 0.608, 0.603, 0.601, 0.607, 0.602, 0.603]])
    baseline_reca = np.vstack([[0.582, 0.594, 0.594, 0.596, 0.592, 0.597, 0.594, 0.604, 0.601], [0.576, 0.596, 0.596, 0.597, 0.594, 0.6, 0.598, 0.601, 0.6]])

    # fix_cifar model
    fixcifar_fid = np.vstack([[5.261, 5.348, 5.779, 5.905, 6.795, 6.084, 8.165, 6.549, 6.807], [5.296, 5.4, 5.419, 5.654, 6.104, 6.524, 6.47, 6.197, 6.629]])
    fixcifar_prec = np.vstack([[0.65, 0.636, 0.623, 0.62, 0.61, 0.617, 0.591, 0.608, 0.601], [0.648, 0.636, 0.627, 0.623, 0.612, 0.607, 0.607, 0.607, 0.603]])
    fixcifar_reca = np.vstack([[0.582, 0.59, 0.594, 0.596, 0.594, 0.596, 0.584, 0.596, 0.601], [0.585, 0.59, 0.599, 0.594, 0.597, 0.597, 0.601, 0.602, 0.6]])

    # mean and error
    baseline_fid_mean, baseline_fid_err = np.mean(baseline_fid, axis=0), np.absolute(baseline_fid[1] - baseline_fid[0]) / 2
    baseline_prec_mean, baseline_prec_err = np.mean(baseline_prec, axis=0), np.absolute(baseline_prec[1] - baseline_prec[0]) / 2
    baseline_reca_mean, baseline_reca_err = np.mean(baseline_reca, axis=0), np.absolute(baseline_reca[1] - baseline_reca[0]) / 2

    fixcifar_fid_mean, fixcifar_fid_err = np.mean(fixcifar_fid, axis=0), np.absolute(fixcifar_fid[1] - fixcifar_fid[0]) / 2
    fixcifar_prec_mean, fixcifar_prec_err = np.mean(fixcifar_prec, axis=0), np.absolute(fixcifar_prec[1] - fixcifar_prec[0]) / 2
    fixcifar_reca_mean, fixcifar_reca_err = np.mean(fixcifar_reca, axis=0), np.absolute(fixcifar_reca[1] - fixcifar_reca[0]) / 2

    with plt.style.context('ggplot'):
        fig, ax1 = plt.subplots(figsize=(8, 8/1.6))
        ax2 = ax1.twinx()

        lns1 = ax1.errorbar(xs, fixcifar_fid_mean, yerr=fixcifar_fid_err, color="red", capsize=3, label="FID $\downarrow$")
        lns2 = ax2.errorbar(xs, fixcifar_prec_mean, yerr=fixcifar_prec_err, color="blue", capsize=3, label=r"Precision $\uparrow$")
        lns3 = ax2.errorbar(xs, fixcifar_reca_mean, yerr=fixcifar_reca_err, color="green", capsize=3, label=r"Recall $\uparrow$")

        lns4 = ax1.errorbar(xs, baseline_fid_mean, yerr=baseline_fid_err, linestyle="--", color="red", capsize=3)
        lns5 = ax2.errorbar(xs, baseline_prec_mean, yerr=baseline_prec_err, linestyle="--", color="blue", capsize=3)
        lns6 = ax2.errorbar(xs, baseline_reca_mean, yerr=baseline_reca_err, linestyle="--", color="green", capsize=3)
            
        ax1.set_xlabel("Number of ImageNet training samples (in thousands)", fontsize=10)
        ax1.set_xticks(xs, xs)
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

        ax1.set_ylabel("FID", fontsize=10)
        ax2.set_ylabel("Precision and Recall", fontsize=10)
        ax1.set_yticks(np.linspace(5.2, 8.2, num=11), np.linspace(5.2, 8.2, num=11))
        ax2.set_yticks(np.linspace(0.55, 0.65, num=11), np.linspace(0.55, 0.65, num=11))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        plt.title("Generation quality of samples from CIFAR domain (50k)", fontsize=10)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2)
            
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-04-06/figures", "cifar_imagenet_quality.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-04-06/figures", "cifar_imagenet_quality.pdf"), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    os.makedirs("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-04-06/figures", exist_ok=True)
    plot_cifar_quality()