import numpy as np 
import os 
import matplotlib.pyplot as plt

def plot_num_dist():
    labels = ["Train", "Generated"]

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    # [0, 0]
    color_vals, color_stds = [99, 99.43], [0.0, 0.02 * 1.96]
    gray_vals, gray_stds = [1, 0.57], [0.0, 0.02 * 1.96]

    rects1 = ax[0, 0].bar(x - width/2, color_vals, yerr=color_stds, width=width, label='Color')
    rects2 = ax[0, 0].bar(x + width/2, gray_vals, yerr=gray_stds, width=width, label='Gray')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[0, 0].set_ylabel('Number of Images (%)')
    ax[0, 0].set_title('99C-01G')
    ax[0, 0].set_xticks(x, labels)
    ax[0, 0].set_ylim(0, 105)
    # ax[0, 0].legend()

    ax[0, 0].bar_label(rects1, label_type="edge", fmt="%.2f")
    ax[0, 0].bar_label(rects2, label_type="edge", fmt="%.2f")

    # [0, 1]
    color_vals, color_stds = [95, 95.52], [0.0, 0.04 * 1.96]
    gray_vals, gray_stds = [5, 4.48], [0.0, 0.04 * 1.96]

    rects1 = ax[0, 1].bar(x - width/2, color_vals, yerr=color_stds, width=width, label='Color')
    rects2 = ax[0, 1].bar(x + width/2, gray_vals, yerr=gray_stds, width=width, label='Gray')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[0, 1].set_ylabel('Number of Images (%)')
    ax[0, 1].set_title('95C-05G')
    ax[0, 1].set_xticks(x, labels)
    ax[0, 1].set_ylim(0, 105)
    # ax[0, 1].legend()

    ax[0, 1].bar_label(rects1, label_type="edge", fmt="%.2f")
    ax[0, 1].bar_label(rects2, label_type="edge", fmt="%.2f")

    # [1, 0]
    color_vals, color_stds = [5, 4.33], [0.0, 0.06 * 1.96]
    gray_vals, gray_stds = [95, 95.67], [0.0, 0.06 * 1.96]

    rects1 = ax[1, 0].bar(x - width/2, color_vals, yerr=color_stds, width=width, label='Color')
    rects2 = ax[1, 0].bar(x + width/2, gray_vals, yerr=gray_stds, width=width, label='Gray')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[1, 0].set_ylabel('Number of Images (%)')
    ax[1, 0].set_title('05C-95G')
    ax[1, 0].set_xticks(x, labels)
    ax[1, 0].set_ylim(0, 105)
    # ax[1, 0].legend()

    ax[1, 0].bar_label(rects1, label_type="edge", fmt="%.2f")
    ax[1, 0].bar_label(rects2, label_type="edge", fmt="%.2f")

    # [1, 1]
    color_vals, color_stds = [1, 0.26], [0.0, 0.02 * 1.96]
    gray_vals, gray_stds = [99, 99.74], [0.0, 0.02 * 1.96]

    rects1 = ax[1, 1].bar(x - width/2, color_vals, yerr=color_stds, width=width, label='Color')
    rects2 = ax[1, 1].bar(x + width/2, gray_vals, yerr=gray_stds, width=width, label='Gray')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[1, 1].set_ylabel('Number of Images (%)')
    ax[1, 1].set_title('01C-99G')
    ax[1, 1].set_xticks(x, labels)
    ax[1, 1].set_ylim(0, 105)
    # ax[1, 1].legend()

    ax[1, 1].bar_label(rects1, label_type="edge", fmt="%.2f")
    ax[1, 1].bar_label(rects2, label_type="edge", fmt="%.2f")

    plt.legend(bbox_to_anchor=(0, -0.05), ncol=2)

    fig.tight_layout()

    plt.show()

    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2022-12-08/cifar10/figures", "num_dist_boostrap.png"), dpi=80)
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2022-12-08/cifar10/figures", "num_dist_boostrap.pdf"), dpi=80)


if __name__ == "__main__":
    plot_num_dist()