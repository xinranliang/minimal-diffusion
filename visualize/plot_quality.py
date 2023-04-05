import numpy as np 
import os 
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def get_args():
    parser = argparse.ArgumentParser("Plot figures about trends in quality metrics")

    parser.add_argument('--dataset', type=str, help="which dataset we use for training and evaluation")
    parser.add_argument('--date', type=str, help="experiment date for logging purpose")

    args = parser.parse_args()
    args.output_dir = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs", args.date, args.dataset, "figures")
    os.makedirs(args.output_dir, exist_ok=True)

    return args 

def plot_fid(fid_arr, title, fig_name):
    xs = [100, 95, 90, 10, 5]

    plt.plot(xs, fid_arr, linestyle="dotted", linewidth=2, color="orange", marker="s", markersize=6, markeredgecolor="darkblue", markerfacecolor="darkblue")
    plt.xlim(105, 0)
    plt.xlabel("Percentage in training distribution (%)")
    plt.xticks(xs, xs)
    plt.ylabel("FID ($\downarrow$)")
    plt.title(title)

    for i, v in enumerate(xs):
        plt.annotate("%.2f" % (fid_arr[i]), xy=(v, fid_arr[i]), xytext=(-10, 10), textcoords="offset points", fontsize=8)

    plt.show()
    plt.savefig(os.path.join(args.output_dir, f"{fig_name}.png"), dpi=100)
    plt.savefig(os.path.join(args.output_dir, f"{fig_name}.pdf"), dpi=100)
    plt.close()


def plot_pre_rec(precision_array, recall_array, title, fig_name):
    xs = [100, 95, 90, 10, 5]

    plt.plot(xs, precision_array, linestyle="dotted", linewidth=2, color="orange", marker="s", markersize=6, markeredgecolor="darkblue", markerfacecolor="darkblue", label="Precision")
    plt.plot(xs, recall_array, linestyle="dotted", linewidth=2, color="orange", marker="s", markersize=6, markeredgecolor="darkgreen", markerfacecolor="darkgreen", label="Recall")

    plt.xlim(105, 0)
    plt.xlabel("Percentage in training distribution (%)")
    plt.xticks(xs, xs)
    plt.ylabel("Value")
    plt.title(title)

    for i, v in enumerate(xs):
        plt.annotate("%.3f" % (precision_array[i]), xy=(v, precision_array[i]), xytext=(-10, -14), textcoords="offset points", fontsize=8)
        plt.annotate("%.3f" % (recall_array[i]), xy=(v, recall_array[i]), xytext=(-10, 10), textcoords="offset points", fontsize=8)

    plt.legend()
    plt.show()
    plt.savefig(os.path.join(args.output_dir, f"{fig_name}.png"), dpi=100)
    plt.savefig(os.path.join(args.output_dir, f"{fig_name}.pdf"), dpi=100)
    plt.close()


# this function stores all quality metrics logs as specified in spreadsheet https://docs.google.com/spreadsheets/d/1eyy_s2LPbZ7aah_rVGqj2dOguEzpASYxTF9dL7eUA1Y/edit?usp=sharing
# order is always from [100%, 95%, 90%, 10%, 5%]
def plot_0122(plot_dir):
    uncond_color_fid = [6.298, 6.364, 6.849, 8.549, 9.872]
    uncond_color_precision = [0.664, 0.66, 0.658, 0.639, 0.618]
    uncond_color_recall = [0.609, 0.619, 0.609, 0.586, 0.599]

    uncond_gray_fid = [4.55, 3.805, 3.995, 7.25, 7.939]
    uncond_gray_precision = [0.736, 0.731, 0.735, 0.727, 0.712]
    uncond_gray_recall = [0.597, 0.6, 0.593, 0.558, 0.576]

    cond_color_fid = [5.787, 5.667, 5.644, 6.782, 8.622]
    cond_color_precision = [0.664, 0.66, 0.666, 0.673, 0.648]
    cond_color_recall = [0.597, 0.605, 0.61, 0.594, 0.577]

    cond_gray_fid = [3.73, 3.745, 3.661, 6.815, 7.554]
    cond_gray_precision = [0.735, 0.741, 0.736, 0.71, 0.717]
    cond_gray_recall = [0.607, 0.596, 0.598, 0.573, 0.56]

    plot_fid(uncond_color_fid, "Color generations from Unconditional diffusion model", "uncond_color_fid")
    plot_fid(cond_color_fid, "Color generations from Conditional diffusion model", "cond_color_fid")
    plot_fid(uncond_gray_fid, "Gray generations from Unconditional diffusion model", "uncond_gray_fid")
    plot_fid(cond_gray_fid, "Gray generations from Conditional diffusion model", "cond_gray_fid")
    
    plot_pre_rec(uncond_color_precision, uncond_color_recall, "Color generations from Unconditional diffusion model", "uncond_color_precision_recall")
    plot_pre_rec(cond_color_precision, cond_color_recall, "Color generations from Conditional diffusion model", "cond_color_precision_recall")
    plot_pre_rec(uncond_gray_precision, uncond_color_recall, "Gray generations from Unconditional diffusion model", "uncond_gray_precision_recall")
    plot_pre_rec(cond_gray_precision, cond_gray_recall, "Gray generations from Conditional diffusion model", "cond_gray_precision_recall")

def plot_guidance(fid_arr, prec_arr, rec_arr, title, output_dir):
    w = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0]

    plt.figure(figsize=(10, 8))
    plt.plot(w, fid_arr, linestyle="-", linewidth=1, color="black", marker="x", markersize=6, markeredgecolor="blue", markerfacecolor="blue")
    
    plt.xlim(-0.5, 4.5)
    plt.xlabel("Scale of Classifier-free Guidance")
    plt.xticks(w, w)
    plt.ylabel("FID")
    plt.title(title)

    # for i, v in enumerate(w):
        # plt.annotate("%.3f" % (fid_arr[i]), xy=(v, fid_arr[i]), xytext=(-10, 10), textcoords="offset points", fontsize=8)

    plt.show()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "guidance_fid.png"), dpi=100)
    plt.savefig(os.path.join(output_dir, "guidance_fid.pdf"), dpi=100)
    plt.close()

    plt.figure(figsize=(10, 8))

    plt.plot(w, prec_arr, linestyle="-", linewidth=1, color="black", marker="x", markersize=6, markeredgecolor="blue", markerfacecolor="blue", label="Precision")
    plt.plot(w, rec_arr, linestyle="-", linewidth=1, color="black", marker="x", markersize=6, markeredgecolor="red", markerfacecolor="red", label="Recall")
    
    plt.xlim(-0.5, 4.5)
    plt.xlabel("Scale of Classifier-free Guidance")
    plt.xticks(w, w)
    plt.ylabel("Precision / Recall")
    plt.title(title)

    # for i, v in enumerate(w):
        # plt.annotate("%.3f" % (prec_arr[i]), xy=(v, prec_arr[i]), xytext=(-10, -14), textcoords="offset points", fontsize=8)
        # plt.annotate("%.3f" % (rec_arr[i]), xy=(v, rec_arr[i]), xytext=(-10, 14), textcoords="offset points", fontsize=8)

    plt.legend()
    plt.show()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "guidance_prec_rec.png"), dpi=100)
    plt.savefig(os.path.join(output_dir, "guidance_prec_rec.pdf"), dpi=100)
    plt.close()


def plot_fix(fid_arr, prec_arr, rec_arr, output_dir, fix, xlabel):
    if fix == "total":
        xs = [100, 95, 90, 70, 50, 30, 10, 5]
    elif fix == "subgroup":
        xs = [0, 2.5, 5, 10, 20, 40, 47.5]

    plt.figure(figsize=(12, 8))
    plt.plot(xs, fid_arr, linestyle="-", linewidth=1, color="black", marker="x", markersize=6, markeredgecolor="blue", markerfacecolor="blue")
    if fix == "total":
        plt.xlim(105, 0)
    elif fix == "subgroup":
        plt.xlim(-0.5, 48)
    plt.xlabel(xlabel)
    plt.xticks(xs, xs)
    plt.ylabel("FID")
    plt.title("Conditional color generation quality w.r.t training distribution")

    # for i, v in enumerate(xs):
        # plt.annotate("%.2f" % (fid_arr[i]), xy=(v, fid_arr[i]), xytext=(-10, 10), textcoords="offset points", fontsize=8)

    plt.show()
    plt.savefig(os.path.join(output_dir, f"fix_{fix}_fid.png"), dpi=100)
    plt.savefig(os.path.join(output_dir, f"fix_{fix}_fid.pdf"), dpi=100)
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(xs, prec_arr, linestyle="-", linewidth=1, color="black", marker="x", markersize=6, markeredgecolor="blue", markerfacecolor="blue", label="Precision")
    plt.plot(xs, rec_arr, linestyle="-", linewidth=1, color="black", marker="x", markersize=6, markeredgecolor="red", markerfacecolor="red", label="Recall")
    if fix == "total":
        plt.xlim(105, 0)
    elif fix == "subgroup":
        plt.xlim(-0.5, 48)
    plt.xlabel(xlabel)
    plt.xticks(xs, xs)
    plt.ylabel("Precision / Recall")
    plt.title("Conditional color generation quality w.r.t training distribution")

    # for i, v in enumerate(xs):
        # plt.annotate("%.2f" % (fid_arr[i]), xy=(v, fid_arr[i]), xytext=(-10, 10), textcoords="offset points", fontsize=8)
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(output_dir, f"fix_{fix}_prec_rec.png"), dpi=100)
    plt.savefig(os.path.join(output_dir, f"fix_{fix}_prec_rec.pdf"), dpi=100)
    plt.close()


def plot_020607():

    # plot effect of guidance
    fid_25 = [18.815, 13.444, 10.817, 9.456, 8.837, 8.739, 10.87, 14.21, 17.618]
    precision_25 = [0.636, 0.685, 0.71, 0.72, 0.733, 0.736, 0.768, 0.772, 0.767]
    recall_25 = [0.427, 0.419, 0.408, 0.414, 0.413, 0.4, 0.361, 0.336, 0.319]

    fid_500 = [5.379, 4.48, 4.424, 4.838, 5.487, 6.329, 11.39, 16.296, 20.705]
    precision_500 = [0.68, 0.707, 0.719, 0.739, 0.748, 0.777, 0.803, 0.797, 0.777]
    recall_500 = [0.597, 0.578, 0.56, 0.545, 0.514, 0.489, 0.402, 0.326, 0.297]

    plot_guidance(fid_25, precision_25, recall_25, 
                    "Conditional generation trained on 2.5K color CIFAR10 images",  
                    os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-02-07/cifar10/color2500_gray0", "figures"))
    plot_guidance(fid_500, precision_500, recall_500, 
                    "Conditional generation trained on 50K color CIFAR10 images",  
                    os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-02-06/cifar10/color1.0_gray0.0", "figures"))

    # plot fix total
    fid_fix_total = [5.379, 5.27, 5.508, 5.94, 5.834, 6.145, 6.609, 7.987]
    prec_fix_total = [0.68, 0.676, 0.681, 0.668, 0.672, 0.668, 0.667, 0.649]
    rec_fix_total = [0.597, 0.599, 0.601, 0.594, 0.595, 0.579, 0.578, 0.568]

    plot_fix(fid_fix_total, prec_fix_total, rec_fix_total, 
                "/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-02-06/cifar10/figures", "total",
                "Percentage in training distribution (%)")

    # plot fix subgroup
    fid_fix_subgroup = [18.815, 26.72, 23.686, 17.354, 9.873, 7.849, 7.987]
    prec_fix_subgroup = [0.636, 0.62, 0.673, 0.689, 0.666, 0.639, 0.649]
    rec_fix_subgroup = [0.427, 0.392, 0.399, 0.459, 0.546, 0.574, 0.568]

    plot_fix(fid_fix_subgroup, prec_fix_subgroup, rec_fix_subgroup, 
                "/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-02-07/cifar10/figures", "subgroup",
                "Number of gray images in training distribution (in thousands)")


def plot_fixtotal():
    xs = [100, 95, 90, 70, 50, 30, 10, 5]

    # plot color 
    fid_color = [[5.379, 5.312, 5.489, 5.946, 5.838, 6.143, 6.612, 7.993], [5.273, 5.394, 5.554, 5.565, 5.6, 5.982, 7.008, 8.132]]
    fid_color = np.vstack(fid_color)
    fid_color_mean, fid_color_err = np.mean(fid_color, axis=0), np.std(fid_color, axis=0) / np.sqrt(2)
    precision_color = [[0.68, 0.669, 0.677, 0.678, 0.672, 0.66, 0.66, 0.66], [0.648, 0.644, 0.642, 0.645, 0.643, 0.64, 0.637, 0.619]]
    precision_color = np.vstack(precision_color)
    precision_color_mean, precision_color_err = np.mean(precision_color, axis=0), np.std(precision_color, axis=0) / np.sqrt(2)
    recall_color = [[0.597, 0.602, 0.602, 0.586, 0.581, 0.593, 0.584, 0.581], [0.585, 0.583, 0.583, 0.581, 0.569, 0.573, 0.553, 0.552]]
    recall_color = np.vstack(recall_color)
    recall_color_mean, recall_color_err = np.mean(recall_color, axis=0), np.std(recall_color, axis=0) / np.sqrt(2)

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        lns1 = ax1.errorbar(xs, fid_color_mean, yerr=fid_color_err, color="red", capsize=3, label="FID $\downarrow$")
        lns2 = ax2.errorbar(xs, precision_color_mean, yerr=precision_color_err, color="blue", capsize=3, label=r"Precision $\uparrow$")
        lns3 = ax2.errorbar(xs, recall_color_mean, yerr=recall_color_err, color="green", capsize=3, label=r"Recall $\uparrow$")
            
        ax1.set_xlabel("Percentage in training distribution ($\%$)", fontsize=10)
        ax1.set_xticks(xs, xs)

        ax1.set_ylabel("FID", fontsize=10)
        ax2.set_ylabel("Precision and Recall", fontsize=10)
        ax1.set_yticks(np.linspace(3.2, 8.2, num=11), np.linspace(3.2, 8.2, num=11))
        ax2.set_yticks(np.linspace(0.5, 0.75, num=11), np.linspace(0.5, 0.75, num=11))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        plt.title("Class-conditional color generation quality (N_total = 50k)", fontsize=10)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2)

    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-03-27/cifar10/figures", "fixtotal_color.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-03-27/cifar10/figures", "fixtotal_color.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

    # plot gray
    fid_gray = [[3.36, 3.809, 3.752, 5.721, 4.363, 5.883, 6.514, 7.42], [3.391, 3.68, 4.396, 3.806, 4.266, 5.302, 6.994, 7.007]]
    fid_gray = np.vstack(fid_gray)
    fid_gray_mean, fid_gray_err = np.mean(fid_gray, axis=0), np.std(fid_gray, axis=0) / np.sqrt(2)
    precision_gray = [[0.739, 0.74, 0.738, 0.743, 0.727, 0.739, 0.735, 0.724], [0.709, 0.712, 0.703, 0.715, 0.71, 0.706, 0.69, 0.681]]
    precision_gray = np.vstack(precision_gray)
    precision_gray_mean, precision_gray_err = np.mean(precision_gray, axis=0), np.std(precision_gray, axis=0) / np.sqrt(2)
    recall_gray = [[0.597, 0.592, 0.595, 0.559, 0.594, 0.567, 0.569, 0.565], [0.577, 0.573, 0.563, 0.57, 0.563, 0.551, 0.536, 0.54]]
    recall_gray = np.vstack(recall_gray)
    recall_gray_mean, recall_gray_err = np.mean(recall_gray, axis=0), np.std(recall_gray, axis=0) / np.sqrt(2)

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        lns1 = ax1.errorbar(xs, fid_gray_mean, yerr=fid_gray_err, color="red", capsize=3, label="FID $\downarrow$")
        lns2 = ax2.errorbar(xs, precision_gray_mean, yerr=precision_gray_err, color="blue", capsize=3, label=r"Precision $\uparrow$")
        lns3 = ax2.errorbar(xs, recall_gray_mean, yerr=recall_gray_err, color="green", capsize=3, label=r"Recall $\uparrow$")
            
        ax1.set_xlabel("Percentage in training distribution ($\%$)", fontsize=10)
        ax1.set_xticks(xs, xs)

        ax1.set_ylabel("FID", fontsize=10)
        ax2.set_ylabel("Precision and Recall", fontsize=10)
        ax1.set_yticks(np.linspace(3.2, 8.2, num=11), np.linspace(3.2, 8.2, num=11))
        ax2.set_yticks(np.linspace(0.5, 0.75, num=11), np.linspace(0.5, 0.75, num=11))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        plt.title("Class-conditional gray generation quality (N_total = 50k)", fontsize=10)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="center")

    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-03-27/cifar10/figures", "fixtotal_gray.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-03-27/cifar10/figures", "fixtotal_gray.pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_fixcolor(num_color):
    if num_color == 15000:
        xs = np.array([0, 5000, 10000, 15000, 20000, 25000, 30000, 35000], dtype=float) / num_color
        fid_color = [[11.819, 10.086, 9.365, 8.799, 8.49, 8.386, 8.281, 8.298], [11.878, 10.111, 9.067, 9.126, 8.773, 8.685, 8.279, 8.262]]
        fid_color = np.vstack(fid_color)
        fid_color_mean, fid_color_err = np.mean(fid_color, axis=0), np.std(fid_color, axis=0)
        precision_color = [[0.684, 0.69, 0.677, 0.676, 0.668, 0.668, 0.668, 0.663], [0.684, 0.689, 0.685, 0.68, 0.674, 0.667, 0.676, 0.661]]
        precision_color = np.vstack(precision_color)
        precision_color_mean, precision_color_err = np.mean(precision_color, axis=0), np.std(precision_color, axis=0) / np.sqrt(2)
        recall_color = [[0.54, 0.56, 0.56, 0.571, 0.582, 0.578, 0.574, 0.583], [0.53, 0.554, 0.561, 0.577, 0.579, 0.582, 0.581, 0.58]]
        recall_color = np.vstack(recall_color)
        recall_color_mean, recall_color_err = np.mean(recall_color, axis=0), np.std(recall_color, axis=0)

        plt.figure(figsize=(8, 8/1.6))
        with plt.style.context('ggplot'):
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()

            lns1 = ax1.errorbar(xs, fid_color_mean, yerr=fid_color_err, color="red", capsize=3, label="FID $\downarrow$")
            lns2 = ax2.errorbar(xs, precision_color_mean, yerr=precision_color_err, color="blue", capsize=3, label=r"Precision $\uparrow$")
            lns3 = ax2.errorbar(xs, recall_color_mean, yerr=recall_color_err, color="green", capsize=3, label=r"Recall $\uparrow$")
            
            ax1.set_xlabel("Ratio of training samples from Gray domain w.r.t. Color domain", fontsize=10)
            ax1.set_xticks(xs, xs)
            ax1.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

            ax1.set_ylabel("FID", fontsize=10)
            ax2.set_ylabel("Precision and Recall", fontsize=10)
            ax1.set_yticks(np.linspace(8, 12, num=11), np.linspace(8, 12, num=11))
            ax2.set_yticks(np.linspace(0.5, 0.7, num=11), np.linspace(0.5, 0.7, num=11))
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            plt.title("Class-conditional color generation quality (N_color = 15k, CIFAR10)", fontsize=10)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="center")
            
        plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-04-01/cifar10/figures", "fixcolor_15k.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-04-01/cifar10/figures", "fixcolor_15k.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    


if __name__ == "__main__":
    args = get_args()
    
    if args.date == "2023-01-22":
        plot_0122(args.output_dir)
    elif args.date == "2023-02-06" or args.date == "2023-02-07":
        plot_020607()
    elif args.date == "2023-03-27":
        # plot_fixtotal()
        plot_fixcolor(num_color = 15000)