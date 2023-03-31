import numpy as np 
import os 
import matplotlib.pyplot as plt

def plot_guidance_dist(color_count, gray_count, color_split, gray_split, num_samples = 50000):
    ws = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    color_count = np.vstack(color_count) / num_samples
    color_count_mean, color_count_err = np.mean(color_count, axis=0), np.std(color_count, axis=0) / np.sqrt(2)

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        plt.errorbar(ws, color_count_mean, yerr=color_count_err, linestyle="-", linewidth=1, color="black", capsize=3, label="Percentage in generated samples")
        plt.axhline(color_split, xmin=ws[0], xmax=ws[-1], linestyle="--", color="gray", label="Percentage in training distribution")
        plt.xticks(ws, ws)
        plt.ylim(0, 1.01)
        plt.yticks(np.arange(0, 1.01, step=0.1))
        plt.xlabel("Scale of classifier-free guidance ($w$)")
        plt.ylabel("Value")
        plt.title("Relationship between Guidance scale and Sampling distribution")
        plt.legend()
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-03-27/cifar10/figures", "guidance_count_color{}_gray{}.png".format(color_split, gray_split)), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-03-27/cifar10/figures", "guidance_count_color{}_gray{}.pdf".format(color_split, gray_split)), dpi=300, bbox_inches="tight")
    plt.close()


def call_guidance_dist():
    # 95C-5G
    color = [49728, 49739, 49788, 49824, 49871, 49888, 49973, 49992, 49999]
    gray = [272, 261, 212, 176, 129, 112, 27, 8, 1]
    # plot_guidance_dist(color, gray, 0.95, 0.05)

    # 90C-10G
    color = [[47355, 47750, 47969, 48243, 48493, 48678, 49417, 49764, 49916], [47564, 47916, 48110, 48481, 48644, 48822, 49483, 49760, 49889]]
    gray = [15635, 14678, 13710, 12337, 11467, 10585, 6797, 3911, 2039]
    plot_guidance_dist(color, gray, 0.9, 0.1)

    # 70C-30G
    color = [[34365, 35322, 36290, 37663, 38533, 39415, 43203, 46089, 47961], [33065, 33818, 34772, 35817, 36403, 37306, 40159, 42536, 44388]]
    gray = [15635, 14678, 13710, 12337, 11467, 10585, 6797, 3911, 2039]
    # plot_guidance_dist(color, gray, 0.7, 0.3)

    # 50C-50G
    color = [[22612, 23120, 23881, 24780, 25528, 26310, 29706, 32870, 35804], [22156, 22726, 23687, 24803, 25603, 26392, 30160, 33350, 36222]]
    gray = [27388, 26880, 26119, 25220, 24472, 23690, 20294, 17130, 14196]
    # plot_guidance_dist(color, gray, 0.5, 0.5)

    # 30C-70G
    color = [[13370, 13889, 14774, 15594, 16404, 17008, 21437, 25826, 30644], [12025, 12477, 12987, 13745, 14144, 14680, 17250, 19308, 21558]]
    gray = [36630, 36111, 35226, 34406, 33596, 32992, 28563, 24174, 19356]
    # plot_guidance_dist(color, gray, 0.3, 0.7)

    # 10C-90G
    color = [[3303, 3421, 3750, 3903, 4120, 4396, 5629, 6739, 8472], [3383, 3492, 3899, 4024, 4280, 4641, 7062, 9760, 11915]]
    gray = [46697, 46579, 46250, 46097, 45880, 45604, 44371, 43261, 41528]
    plot_guidance_dist(color, gray, 0.1, 0.9)

    # 5C-95G
    color = [1257, 1300, 1458, 1532, 1555, 1663, 2186, 2683, 3539]
    gray = [48743, 48700, 48542, 48468, 48445, 48337, 47814, 47317, 46461]
    # plot_guidance_dist(color, gray, 0.05, 0.95)



if __name__ == "__main__":
    # plot_num_dist()
    call_guidance_dist()