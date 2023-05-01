import numpy as np 
import os 
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def plot_half_quality(num_images, mode):
    xs = np.array([10000, 20000, 30000, 45000, 60000, 90000, 120000], dtype=int) // 1000

    if num_images == 10000:
        if mode == "color":

            # baseline model
            baseline_fid = np.array([21.527, 13.327, 10.844, 10.085, 9.942, 9.815, 9.857], dtype=float)
            baseline_prec = np.array([0.717, 0.713, 0.69, 0.683, 0.668, 0.662, 0.666], dtype=float)
            baseline_reca = np.array([0.473, 0.538, 0.567, 0.585, 0.602, 0.605, 0.596], dtype=float)

            # half/half model
            half_fid = np.array([15.746, 10.719, 10.36, 10.065, 10.285, 10.389, 10.513], dtype=float)
            half_prec = np.array([0.717, 0.691, 0.67, 0.672, 0.665, 0.656, 0.654], dtype=float)
            half_reca = np.array([0.517, 0.569, 0.593, 0.581, 0.583, 0.597, 0.595], dtype=float)

            with plt.style.context('ggplot'):
                fig, ax1 = plt.subplots(figsize=(8, 8/1.6))
                ax2 = ax1.twinx()

                lns1 = ax1.errorbar(xs, half_fid, color="red", capsize=3, label="FID $\downarrow$")
                lns2 = ax2.errorbar(xs, half_prec, color="blue", capsize=3, label=r"Precision $\uparrow$")
                lns3 = ax2.errorbar(xs, half_reca, color="green", capsize=3, label=r"Recall $\uparrow$")

                lns4 = ax1.errorbar(xs, baseline_fid, linestyle="--", color="red", capsize=3)
                lns5 = ax2.errorbar(xs, baseline_prec, linestyle="--", color="blue", capsize=3)
                lns6 = ax2.errorbar(xs, baseline_reca, linestyle="--", color="green", capsize=3)
                    
                ax1.set_xlabel("$N_{gray}$ (in thousands)", fontsize=10)
                ax1.set_xticks(xs, xs)
                ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

                ax1.set_ylabel("FID", fontsize=10)
                ax2.set_ylabel("Precision and Recall", fontsize=10)
                ax1.set_yticks(np.linspace(9, 22, num=11), np.linspace(9, 22, num=11))
                ax2.set_yticks(np.linspace(0.47, 0.72, num=11), np.linspace(0.47, 0.72, num=11))
                ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                plt.title("Generation quality of Color samples (10k)", fontsize=10)

                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2)
                    
            plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-04-02/mix-cifar10-imagenet/figures", "half_color10k_quality.png"), dpi=300, bbox_inches="tight")
            plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-04-02/mix-cifar10-imagenet/figures", "half_color10k_quality.pdf"), dpi=300, bbox_inches="tight")
            plt.close()
        
        elif mode == "gray":

            # baseline model
            baseline_fid = np.array([27.487, 14.266, 9.125, 7.389, 7.155, 7.31, 7.101], dtype=float)
            baseline_prec = np.array([0.771, 0.763, 0.75, 0.732, 0.732, 0.722, 0.719], dtype=float)
            baseline_reca = np.array([0.449, 0.534, 0.564, 0.594, 0.606, 0.607, 0.613], dtype=float)

            # half/half model
            half_fid = np.array([13.762, 8.467, 8.731, 8.111, 8.354, 8.943, 8.729], dtype=float)
            half_prec = np.array([0.759, 0.737, 0.722, 0.714, 0.717, 0.7, 0.719], dtype=float)
            half_reca = np.array([0.521, 0.577, 0.582, 0.598, 0.603, 0.605, 0.602], dtype=float)

            with plt.style.context('ggplot'):
                fig, ax1 = plt.subplots(figsize=(8, 8/1.6))
                ax2 = ax1.twinx()

                lns1 = ax1.errorbar(xs, half_fid, color="red", capsize=3, label="FID $\downarrow$")
                lns2 = ax2.errorbar(xs, half_prec, color="blue", capsize=3, label=r"Precision $\uparrow$")
                lns3 = ax2.errorbar(xs, half_reca, color="green", capsize=3, label=r"Recall $\uparrow$")

                lns4 = ax1.errorbar(xs, baseline_fid, linestyle="--", color="red", capsize=3)
                lns5 = ax2.errorbar(xs, baseline_prec, linestyle="--", color="blue", capsize=3)
                lns6 = ax2.errorbar(xs, baseline_reca, linestyle="--", color="green", capsize=3)
                    
                ax1.set_xlabel("$N_{color}$ (in thousands)", fontsize=10)
                ax1.set_xticks(xs, xs)
                ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

                ax1.set_ylabel("FID", fontsize=10)
                ax2.set_ylabel("Precision and Recall", fontsize=10)
                ax1.set_yticks(np.linspace(7.0, 27.5, num=11), np.linspace(7.0, 27.5, num=11))
                ax2.set_yticks(np.linspace(0.4, 0.8, num=11), np.linspace(0.4, 0.8, num=11))
                ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                plt.title("Generation quality of Gray samples (10k)", fontsize=10)

                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2)
                    
            plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-04-02/mix-cifar10-imagenet/figures", "half_gray10k_quality.png"), dpi=300, bbox_inches="tight")
            plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-04-02/mix-cifar10-imagenet/figures", "half_gray10k_quality.pdf"), dpi=300, bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    plot_half_quality(num_images=10000, mode="color")
    plot_half_quality(num_images=10000, mode="gray")