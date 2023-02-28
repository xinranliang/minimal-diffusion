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


def plot_guidance_dist(color_count, gray_count, color_split, gray_split, num_samples = 50000):
    w = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    # w = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0], dtype=np.float64)

    plt.figure(figsize=(10, 8))

    plt.bar(w, np.array(color_count) / num_samples, 0.1, label="Portion of color images in samples")
    plt.axhline(color_split, xmin=w[0], xmax=w[-1], linestyle="--", color="blue", label="Portion of color images in training dataset")

    plt.xticks(w, w)
    plt.ylim(0, 1.01)
    plt.yticks(np.arange(0, 1.01, step=0.1))
    plt.xlabel("Scale of classifier-free guidance ($w$)")
    plt.ylabel("Value")
    plt.title("Relationship between Guidance scale and Sampling distribution trained with Color{}-Gray{}".format(color_split, gray_split))
    plt.legend()
    plt.show()

    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-02-06/cifar10/figures", "guidance_count_color{}_gray{}.png".format(color_split, gray_split)), dpi=80)
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-02-06/cifar10/figures", "guidance_count_color{}_gray{}.pdf".format(color_split, gray_split)), dpi=80)


def call_guidance_dist():
    # 95C-5G
    color = [47157, 47531, 47751, 47939, 48112, 48344, 48862, 49236, 49414]
    gray = [2843, 2469, 2249, 2061, 1888, 1656, 1138, 764, 586]
    # plot_guidance_dist(color, gray, 0.95, 0.05)

    # 90C-10G
    color = [43872, 44394, 44754, 45254, 45495, 46007, 47115, 47805, 48202]
    gray = [6128, 5606, 5246, 4746, 4505, 3993, 2885, 2195, 1798]
    # plot_guidance_dist(color, gray, 0.9, 0.1)

    # 70C-30G
    color = [32754, 33753, 34459, 35455, 36123, 37067, 39924, 42055, 43579]
    gray = [17246, 16247, 15541, 14545, 13877, 12933, 10076, 7945, 6421]
    plot_guidance_dist(color, gray, 0.7, 0.3)

    # 50C-50G
    color = [22443, 23007, 23643, 24595, 25320, 26160, 29409, 32146, 34792]
    gray = [12006, 12507, 13012, 13906, 14442, 15158, 18130, 20866, 23689]
    # plot_guidance_dist(color, gray, 0.5, 0.5)

    # 30C-70G
    color = [12006, 12507, 13012, 13906, 14442, 15158, 18130, 20866, 23689]
    gray = [37994, 37493, 36988, 36094, 35558, 34842, 31870, 29134, 26311]
    # plot_guidance_dist(color, gray, 0.3, 0.7)

    # 10C-90G
    color = [3314, 3443, 3654, 3858, 4146, 4370, 5628, 6613, 8100]
    gray = [46686, 46557, 46346, 46142, 45854, 45630, 44372, 43387, 41900]
    # plot_guidance_dist(color, gray, 0.1, 0.9)

    # 5C-95G
    color = [1221, 1338, 1358, 1522, 1604, 1694, 2209, 2716, 3383]
    gray = [48779, 48662, 48642, 48478, 48396, 48306, 47791, 47284, 46617]
    plot_guidance_dist(color, gray, 0.05, 0.95)



if __name__ == "__main__":
    # plot_num_dist()
    call_guidance_dist()