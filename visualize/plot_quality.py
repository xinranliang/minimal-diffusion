import numpy as np 
import os 
import argparse
import matplotlib.pyplot as plt


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
    plt.ylabel("FID")
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


if __name__ == "__main__":
    args = get_args()
    
    if args.date == "2023-01-22":
        plot_0122(args.output_dir)