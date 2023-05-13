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
            baseline_fid = np.vstack([[21.527, 13.327, 10.844, 10.085, 9.942, 9.815, 9.857], [22.803, 13.125, 10.723, 10.515, 10.072, 10.005, 9.817]])
            baseline_prec = np.vstack([[0.717, 0.713, 0.69, 0.683, 0.668, 0.662, 0.666], [0.696, 0.693, 0.677, 0.66, 0.661, 0.654, 0.656]])
            baseline_reca = np.vstack([[0.473, 0.538, 0.567, 0.585, 0.602, 0.605, 0.596], [0.468, 0.543, 0.578, 0.591, 0.603, 0.609, 0.602]])

            baseline_fid_mean, baseline_fid_err = np.mean(baseline_fid, axis=0), np.absolute(baseline_fid[1] - baseline_fid[0]) / 2
            baseline_prec_mean, baseline_prec_err = np.mean(baseline_prec, axis=0), np.absolute(baseline_prec[1] - baseline_prec[0]) / 2
            baseline_reca_mean, baseline_reca_err = np.mean(baseline_reca, axis=0), np.absolute(baseline_reca[1] - baseline_reca[0]) / 2

            # half/half model
            half_fid = np.vstack([[15.746, 10.719, 10.36, 10.065, 10.285, 10.389, 10.513], [16.351, 10.813, 10.028, 10.618, 12.519, 10.137, 10.314]])
            half_prec = np.vstack([[0.717, 0.691, 0.67, 0.672, 0.665, 0.656, 0.654], [0.707, 0.671, 0.665, 0.651, 0.646, 0.653, 0.647]])
            half_reca = np.vstack([[0.517, 0.569, 0.593, 0.581, 0.583, 0.597, 0.595], [0.523, 0.578, 0.596, 0.6, 0.603, 0.6, 0.605]])

            half_fid_mean, half_fid_err = np.mean(half_fid, axis=0), np.absolute(half_fid[1] - half_fid[0]) / 2
            half_prec_mean, half_prec_err = np.mean(half_prec, axis=0), np.absolute(half_prec[1] - half_prec[0]) / 2
            half_reca_mean, half_reca_err = np.mean(half_reca, axis=0), np.absolute(half_reca[1] - half_reca[0]) / 2

            with plt.style.context('ggplot'):
                fig, ax1 = plt.subplots(figsize=(8, 8/1.6))
                ax2 = ax1.twinx()

                lns1 = ax1.errorbar(xs, half_fid_mean, yerr=half_fid_err, color="red", capsize=3, label="FID $\downarrow$")
                lns2 = ax2.errorbar(xs, half_prec_mean, yerr=half_prec_err, color="blue", capsize=3, label=r"Precision $\uparrow$")
                lns3 = ax2.errorbar(xs, half_reca_mean, yerr=half_reca_err, color="green", capsize=3, label=r"Recall $\uparrow$")

                lns4 = ax1.errorbar(xs, baseline_fid_mean, yerr=baseline_fid_err, linestyle="--", color="red", capsize=3)
                lns5 = ax2.errorbar(xs, baseline_prec_mean, yerr=baseline_prec_err, linestyle="--", color="blue", capsize=3)
                lns6 = ax2.errorbar(xs, baseline_reca_mean, yerr=baseline_reca_err, linestyle="--", color="green", capsize=3)
                    
                ax1.set_xlabel("$N_{gray}$ (in thousands)", fontsize=10)
                ax1.set_xticks(xs, xs)
                ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

                ax1.set_ylabel("FID", fontsize=10)
                ax2.set_ylabel("Precision and Recall", fontsize=10)
                ax1.set_yticks(np.linspace(9, 23, num=11), np.linspace(9, 23, num=11))
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
            baseline_fid = np.vstack([[27.487, 14.266, 9.125, 7.389, 7.155, 7.31, 7.101], [27.941, 14.346, 9.68, 7.26, 6.896, 6.896, 6.509]])
            baseline_prec = np.vstack([[0.771, 0.763, 0.75, 0.732, 0.732, 0.722, 0.719], [0.777, 0.768, 0.749, 0.731, 0.722, 0.717, 0.712]])
            baseline_reca = np.vstack([[0.449, 0.534, 0.564, 0.594, 0.606, 0.607, 0.613], [0.44, 0.522, 0.558, 0.601, 0.596, 0.613, 0.615]])

            baseline_fid_mean, baseline_fid_err = np.mean(baseline_fid, axis=0), np.absolute(baseline_fid[1] - baseline_fid[0]) / 2
            baseline_prec_mean, baseline_prec_err = np.mean(baseline_prec, axis=0), np.absolute(baseline_prec[1] - baseline_prec[0]) / 2
            baseline_reca_mean, baseline_reca_err = np.mean(baseline_reca, axis=0), np.absolute(baseline_reca[1] - baseline_reca[0]) / 2

            # half/half model
            half_fid = np.vstack([[13.762, 8.467, 8.731, 8.111, 8.354, 8.943, 8.729], [13.227, 8.611, 7.691, 8.705, 9.68, 8.102, 8.156]])
            half_prec = np.vstack([[0.759, 0.737, 0.722, 0.714, 0.717, 0.7, 0.719], [0.758, 0.727, 0.723, 0.708, 0.704, 0.707, 0.711]])
            half_reca = np.vstack([[0.521, 0.577, 0.582, 0.598, 0.603, 0.605, 0.602], [0.517, 0.582, 0.592, 0.59, 0.595, 0.611, 0.602]])

            half_fid_mean, half_fid_err = np.mean(half_fid, axis=0), np.absolute(half_fid[1] - half_fid[0]) / 2
            half_prec_mean, half_prec_err = np.mean(half_prec, axis=0), np.absolute(half_prec[1] - half_prec[0]) / 2
            half_reca_mean, half_reca_err = np.mean(half_reca, axis=0), np.absolute(half_reca[1] - half_reca[0]) / 2

            with plt.style.context('ggplot'):
                fig, ax1 = plt.subplots(figsize=(8, 8/1.6))
                ax2 = ax1.twinx()

                lns1 = ax1.errorbar(xs, half_fid_mean, yerr=half_fid_err, color="red", capsize=3, label="FID $\downarrow$")
                lns2 = ax2.errorbar(xs, half_prec_mean, yerr=half_prec_err, color="blue", capsize=3, label=r"Precision $\uparrow$")
                lns3 = ax2.errorbar(xs, half_reca_mean, yerr=half_reca_err, color="green", capsize=3, label=r"Recall $\uparrow$")

                lns4 = ax1.errorbar(xs, baseline_fid_mean, yerr=baseline_fid_err, linestyle="--", color="red", capsize=3)
                lns5 = ax2.errorbar(xs, baseline_prec_mean, yerr=baseline_prec_err, linestyle="--", color="blue", capsize=3)
                lns6 = ax2.errorbar(xs, baseline_reca_mean, yerr=baseline_reca_err, linestyle="--", color="green", capsize=3)
                    
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
    

    elif num_images == 120000:
        if mode == "color":

            # baseline model
            baseline_fid = np.vstack([[17.449, 8.624, 6.128, 5.239, 5.075, 4.848, 4.972], [18.538, 8.699, 6.278, 5.809, 5.275, 5.255, 5.153]])
            baseline_prec = np.vstack([[0.67, 0.668, 0.649, 0.638, 0.627, 0.625, 0.621], [0.67, 0.659, 0.643, 0.628, 0.624, 0.617, 0.619]])
            baseline_reca = np.vstack([[0.401, 0.49, 0.532, 0.555, 0.57, 0.577, 0.576], [0.398, 0.497, 0.536, 0.556, 0.569, 0.576, 0.576]])

            baseline_fid_mean, baseline_fid_err = np.mean(baseline_fid, axis=0), np.absolute(baseline_fid[1] - baseline_fid[0]) / 2
            baseline_prec_mean, baseline_prec_err = np.mean(baseline_prec, axis=0), np.absolute(baseline_prec[1] - baseline_prec[0]) / 2
            baseline_reca_mean, baseline_reca_err = np.mean(baseline_reca, axis=0), np.absolute(baseline_reca[1] - baseline_reca[0]) / 2

            # half/half model
            half_fid = np.vstack([[11.131, 5.961, 5.531, 5.215, 5.3, 5.38, 5.61], [11.824, 6.197, 5.306, 5.683, 7.755, 5.462, 5.506]])
            half_prec = np.vstack([[0.667, 0.643, 0.629, 0.621, 0.619, 0.612, 0.611], [0.667, 0.642, 0.631, 0.618, 0.608, 0.612, 0.613]])
            half_reca = np.vstack([[0.466, 0.535, 0.559, 0.567, 0.57, 0.576, 0.573], [0.467, 0.538, 0.554, 0.566, 0.569, 0.572, 0.571]])

            half_fid_mean, half_fid_err = np.mean(half_fid, axis=0), np.absolute(half_fid[1] - half_fid[0]) / 2
            half_prec_mean, half_prec_err = np.mean(half_prec, axis=0), np.absolute(half_prec[1] - half_prec[0]) / 2
            half_reca_mean, half_reca_err = np.mean(half_reca, axis=0), np.absolute(half_reca[1] - half_reca[0]) / 2

            with plt.style.context('ggplot'):
                fig, ax1 = plt.subplots(figsize=(8, 8/1.6))
                ax2 = ax1.twinx()

                lns1 = ax1.errorbar(xs, half_fid_mean, yerr=half_fid_err, color="red", capsize=3, label="FID $\downarrow$")
                lns2 = ax2.errorbar(xs, half_prec_mean, yerr=half_prec_err, color="blue", capsize=3, label=r"Precision $\uparrow$")
                lns3 = ax2.errorbar(xs, half_reca_mean, yerr=half_reca_err, color="green", capsize=3, label=r"Recall $\uparrow$")

                lns4 = ax1.errorbar(xs, baseline_fid_mean, yerr=baseline_fid_err, linestyle="--", color="red", capsize=3)
                lns5 = ax2.errorbar(xs, baseline_prec_mean, yerr=baseline_prec_err, linestyle="--", color="blue", capsize=3)
                lns6 = ax2.errorbar(xs, baseline_reca_mean, yerr=baseline_reca_err, linestyle="--", color="green", capsize=3)
                    
                ax1.set_xlabel("$N_{gray}$ (in thousands)", fontsize=10)
                ax1.set_xticks(xs, xs)
                ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

                ax1.set_ylabel("FID", fontsize=10)
                ax2.set_ylabel("Precision and Recall", fontsize=10)
                ax1.set_yticks(np.linspace(4.5, 18.5, num=11), np.linspace(4.5, 18.5, num=11))
                ax2.set_yticks(np.linspace(0.4, 0.7, num=11), np.linspace(0.4, 0.7, num=11))
                ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                plt.title("Generation quality of Color samples (120k)", fontsize=10)

                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2)
                    
            plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-04-02/mix-cifar10-imagenet/figures", "half_color120k_quality.png"), dpi=300, bbox_inches="tight")
            plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-04-02/mix-cifar10-imagenet/figures", "half_color120k_quality.pdf"), dpi=300, bbox_inches="tight")
            plt.close()
        
        elif mode == "gray":

            # baseline model
            baseline_fid = np.vstack([[23.389, 9.966, 4.834, 2.925, 2.734, 2.938, 2.631], [23.225, 9.973, 5.096, 2.87, 2.581, 2.617, 2.109]])
            baseline_prec = np.vstack([[0.73, 0.722, 0.712, 0.697, 0.685, 0.68, 0.677], [0.732, 0.723, 0.71, 0.696, 0.687, 0.681, 0.678]])
            baseline_reca = np.vstack([[0.396, 0.485, 0.527, 0.555, 0.569, 0.575, 0.585], [0.387, 0.481, 0.527, 0.557, 0.568, 0.578, 0.586]])

            baseline_fid_mean, baseline_fid_err = np.mean(baseline_fid, axis=0), np.absolute(baseline_fid[1] - baseline_fid[0]) / 2
            baseline_prec_mean, baseline_prec_err = np.mean(baseline_prec, axis=0), np.absolute(baseline_prec[1] - baseline_prec[0]) / 2
            baseline_reca_mean, baseline_reca_err = np.mean(baseline_reca, axis=0), np.absolute(baseline_reca[1] - baseline_reca[0]) / 2

            # half/half model
            half_fid = np.vstack([[9.382, 4.098, 4.43, 3.727, 3.942, 4.649, 4.353], [8.71, 4.453, 3.508, 4.597, 5.52, 3.832, 3.904]])
            half_prec = np.vstack([[0.716, 0.699, 0.68, 0.678, 0.674, 0.668, 0.671], [0.719, 0.69, 0.687, 0.675, 0.666, 0.674, 0.672]])
            half_reca = np.vstack([[0.465, 0.534, 0.547, 0.566, 0.567, 0.564, 0.569], [0.467, 0.533, 0.555, 0.556, 0.559, 0.57, 0.57]])

            half_fid_mean, half_fid_err = np.mean(half_fid, axis=0), np.absolute(half_fid[1] - half_fid[0]) / 2
            half_prec_mean, half_prec_err = np.mean(half_prec, axis=0), np.absolute(half_prec[1] - half_prec[0]) / 2
            half_reca_mean, half_reca_err = np.mean(half_reca, axis=0), np.absolute(half_reca[1] - half_reca[0]) / 2

            with plt.style.context('ggplot'):
                fig, ax1 = plt.subplots(figsize=(8, 8/1.6))
                ax2 = ax1.twinx()

                lns1 = ax1.errorbar(xs, half_fid_mean, yerr=half_fid_err, color="red", capsize=3, label="FID $\downarrow$")
                lns2 = ax2.errorbar(xs, half_prec_mean, yerr=half_prec_err, color="blue", capsize=3, label=r"Precision $\uparrow$")
                lns3 = ax2.errorbar(xs, half_reca_mean, yerr=half_reca_err, color="green", capsize=3, label=r"Recall $\uparrow$")

                lns4 = ax1.errorbar(xs, baseline_fid_mean, yerr=baseline_fid_err, linestyle="--", color="red", capsize=3)
                lns5 = ax2.errorbar(xs, baseline_prec_mean, yerr=baseline_prec_err, linestyle="--", color="blue", capsize=3)
                lns6 = ax2.errorbar(xs, baseline_reca_mean, yerr=baseline_reca_err, linestyle="--", color="green", capsize=3)
                    
                ax1.set_xlabel("$N_{color}$ (in thousands)", fontsize=10)
                ax1.set_xticks(xs, xs)
                ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

                ax1.set_ylabel("FID", fontsize=10)
                ax2.set_ylabel("Precision and Recall", fontsize=10)
                ax1.set_yticks(np.linspace(2.5, 23.5, num=11), np.linspace(2.5, 23.5, num=11))
                ax2.set_yticks(np.linspace(0.35, 0.75, num=11), np.linspace(0.35, 0.75, num=11))
                ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                plt.title("Generation quality of Gray samples (120k)", fontsize=10)

                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
                    
            plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-04-02/mix-cifar10-imagenet/figures", "half_gray120k_quality.png"), dpi=300, bbox_inches="tight")
            plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-04-02/mix-cifar10-imagenet/figures", "half_gray120k_quality.pdf"), dpi=300, bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    # plot_half_quality(num_images=10000, mode="color")
    plot_half_quality(num_images=120000, mode="color")
    plot_half_quality(num_images=120000, mode="gray")