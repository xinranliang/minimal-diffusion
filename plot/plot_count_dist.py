import numpy as np 
import os 
import matplotlib.pyplot as plt

CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

def plot_guidance_dist(color_count, color_split, num_samples = 50000):
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
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-03-27/cifar10/figures", f"guidance_count_color{color_split}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-03-27/cifar10/figures", f"guidance_count_color{color_split}.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_guidance_value(value_70, value_50, value_30):
    ws = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    value_70_mean, value_70_err = np.mean(value_70, axis=0), np.std(value_70, axis=0)
    value_50_mean, value_50_err = np.mean(value_50, axis=0), np.std(value_50, axis=0)
    value_30_mean, value_30_err = np.mean(value_30, axis=0), np.std(value_30, axis=0)

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        plt.errorbar(ws, value_70_mean, yerr=value_70_err, linestyle="-", linewidth=1, color="black", capsize=3, label="70% color training samples")
        plt.errorbar(ws, value_50_mean, yerr=value_50_err, linestyle="-", linewidth=1, color="purple", capsize=3, label="50% color training samples")
        plt.errorbar(ws, value_30_mean, yerr=value_30_err, linestyle="-", linewidth=1, color="blue", capsize=3, label="30% color training samples")
        plt.xticks(ws, ws)
        plt.xlabel("Scale of classifier-free guidance ($w$)")
        plt.ylabel("RGB channel std")
        plt.title("Relationship between Guidance scale and Samples")
        plt.legend()
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-03-27/cifar10/figures", "guidance_channel_std_value.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-03-27/cifar10/figures", "guidance_channel_std_value.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_guidance_classwise(count_dict, group_count, ref_ratio, num_classes=10, num_samples=50000):
    ws = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    color_list = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "navy", "cyan"]

    # list of num_classes tuples (mean, std)
    class_stats = []
    for idx in range(num_classes):
        class_stats_mean, class_stats_std = np.mean(count_dict[CLASSES[idx]], axis=0), np.std(count_dict[CLASSES[idx]], axis=0) / np.sqrt(2)
        class_stats.append((class_stats_mean, class_stats_std))
    assert len(class_stats) == num_classes

    group_count = np.vstack(group_count) / num_samples
    group_count_mean, group_count_err = np.mean(group_count, axis=0), np.std(group_count, axis=0) / np.sqrt(2)

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        # each class
        for idx in range(num_classes):
            class_stats_mean, class_stats_std = class_stats[idx]
            plt.errorbar(ws, class_stats_mean, yerr=class_stats_std, linestyle="-", linewidth=2, capsize=3, alpha=0.5, color=color_list[idx], label=CLASSES[idx])
        # overall mean
        plt.errorbar(ws, group_count_mean, yerr=group_count_err, linestyle="-", linewidth=2, color="black", capsize=3, label="% in synthetic")
        # reference in training distribution
        plt.axhline(ref_ratio, xmin=ws[0], xmax=ws[-1], linestyle="--", color="violet", label="% in real")

        plt.xticks(ws, ws)
        plt.ylim(ref_ratio - 0.2, min(ref_ratio + 0.51, 1.01))
        plt.xlabel("Scale of classifier-free guidance ($w$)")
        plt.ylabel("% in synthetic samples")
        plt.title("Sampled colored distribution w.r.t Guidance scale")
        plt.legend(ncols=4)
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-03-27/cifar10/figures", f"guidance_count_classwise_color{ref_ratio}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-03-27/cifar10/figures", f"guidance_count_classwise_color{ref_ratio}.pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def call_guidance_dist_classwise():
    # 30C-70G
    color = [[12002, 12399, 13225, 13867, 14553, 15075, 18263, 20895, 23620], [12004, 12442, 12942, 13693, 14082, 14618, 17131, 19095, 21139]]
    classwise_dist = {
        CLASSES[0]: [[0.2451, 0.2181, 0.2756, 0.3156, 0.3255, 0.2845, 0.4033, 0.5447, 0.4355], [0.2539, 0.2282, 0.2706, 0.3039, 0.3211, 0.2469, 0.3439, 0.4738, 0.3625]], 
        CLASSES[1]: [[0.2286, 0.2672, 0.2705, 0.2510, 0.3400, 0.3613, 0.4596, 0.5481, 0.5520], [0.2205, 0.2737, 0.2711, 0.2316, 0.3355, 0.3580, 0.4142, 0.5105, 0.4536]],
        CLASSES[2]: [[0.2398, 0.2413, 0.2544, 0.2874, 0.2638, 0.3347, 0.3398, 0.3947, 0.2840], [0.2306, 0.2325, 0.2403, 0.2651, 0.2558, 0.3335, 0.2854, 0.3578, 0.3209]],
        CLASSES[3]: [[0.2558, 0.2373, 0.2730, 0.2933, 0.2679, 0.3225, 0.3527, 0.4282, 0.6219], [0.2576, 0.2537, 0.2884, 0.3075, 0.2355, 0.2835, 0.3373, 0.3742, 0.5153]],
        CLASSES[4]: [[0.2372, 0.2589, 0.2702, 0.2891, 0.2414, 0.3045, 0.4409, 0.3998, 0.5932], [0.2489, 0.2694, 0.2505, 0.2812, 0.2601, 0.2941, 0.4280, 0.3275, 0.5551]],
        CLASSES[5]: [[0.2498, 0.2492, 0.2703, 0.2482, 0.3055, 0.2790, 0.4001, 0.4612, 0.4392], [0.2412, 0.2655, 0.2472, 0.2394, 0.2863, 0.2812, 0.3616, 0.3958, 0.3841]],
        CLASSES[6]: [[0.2262, 0.2598, 0.2443, 0.2542, 0.2925, 0.3184, 0.3183, 0.4962, 0.4572], [0.2139, 0.2388, 0.2494, 0.2650, 0.3087, 0.3426, 0.2979, 0.4780, 0.3885]],
        CLASSES[7]: [[0.2393, 0.2555, 0.2498, 0.2344, 0.2694, 0.2479, 0.2545, 0.2745, 0.4569], [0.2453, 0.2388, 0.2596, 0.2613, 0.2736, 0.2625, 0.2858, 0.3032, 0.3797]],
        CLASSES[8]: [[0.2425, 0.2516, 0.2960, 0.2861, 0.2929, 0.3151, 0.4156, 0.2564, 0.2559], [0.2550, 0.2565, 0.2954, 0.2659, 0.2672, 0.2876, 0.4097, 0.2767, 0.2857]],
        CLASSES[9]: [[0.2359, 0.2412, 0.2411, 0.3141, 0.3130, 0.2504, 0.2709, 0.3831, 0.6330], [0.2336, 0.2321, 0.2162, 0.3176, 0.2751, 0.2370, 0.2657, 0.3269, 0.5859]]
    }
    plot_guidance_classwise(classwise_dist, color, 0.3)

    # 50C-50G
    color = [[22076, 22636, 23582, 24674, 25462, 26271, 29939, 33002, 35650], [22535, 23023, 23775, 24658, 25395, 26157, 29422, 32394, 34909]]
    classwise_dist = {
        CLASSES[0]: [[0.4653, 0.4078, 0.4856, 0.5318, 0.5549, 0.5060, 0.5748, 0.7386, 0.7081], [0.4441, 0.4144, 0.4696, 0.5572, 0.5825, 0.4918, 0.6184, 0.7965, 0.6999]], 
        CLASSES[1]: [[0.4539, 0.4973, 0.4953, 0.4791, 0.5583, 0.5824, 0.6627, 0.7786, 0.6940], [0.4227, 0.4908, 0.5093, 0.4580, 0.5517, 0.6120, 0.7069, 0.7739, 0.7549]],
        CLASSES[2]: [[0.4553, 0.4463, 0.4612, 0.4862, 0.4834, 0.5901, 0.5746, 0.6472, 0.5485], [0.4412, 0.4401, 0.4453, 0.4896, 0.4877, 0.5845, 0.5596, 0.6089, 0.5942]],
        CLASSES[3]: [[0.4788, 0.4471, 0.5034, 0.5207, 0.4879, 0.5525, 0.5875, 0.6545, 0.7827], [0.4688, 0.4496, 0.5059, 0.5227, 0.4690, 0.5456, 0.5633, 0.6685, 0.8473]],
        CLASSES[4]: [[0.4484, 0.4794, 0.4803, 0.5026, 0.4465, 0.5336, 0.6887, 0.6495, 0.8200], [0.4454, 0.4826, 0.4658, 0.4897, 0.4664, 0.5093, 0.6879, 0.6342, 0.7996]],
        CLASSES[5]: [[0.4510, 0.4786, 0.4633, 0.4569, 0.5174, 0.4912, 0.6263, 0.6211, 0.6959], [0.4544, 0.4709, 0.4683, 0.4444, 0.5038, 0.4912, 0.6301, 0.6742, 0.6440]],
        CLASSES[6]: [[0.4386, 0.4719, 0.4302, 0.4514, 0.5467, 0.5588, 0.5146, 0.7393, 0.6697], [0.4130, 0.4521, 0.4364, 0.4667, 0.5423, 0.5669, 0.5455, 0.7234, 0.7056]],
        CLASSES[7]: [[0.4479, 0.4612, 0.4526, 0.4312, 0.4722, 0.4496, 0.4886, 0.5327, 0.6199], [0.4566, 0.4523, 0.4622, 0.4579, 0.4808, 0.4815, 0.5193, 0.5640, 0.6769]],
        CLASSES[8]: [[0.4563, 0.4580, 0.5198, 0.5199, 0.5248, 0.5031, 0.6528, 0.5532, 0.6042], [0.4571, 0.4462, 0.5146, 0.5043, 0.5130, 0.5155, 0.6458, 0.5412, 0.5701]],
        CLASSES[9]: [[0.4109, 0.4576, 0.4637, 0.5530, 0.4893, 0.4687, 0.5166, 0.5713, 0.8432], [0.4114, 0.4291, 0.4396, 0.5444, 0.4989, 0.4605, 0.5136, 0.6212, 0.8414]]
    }
    plot_guidance_classwise(classwise_dist, color, 0.5)

    # 70C-30G
    color = [[32848, 33675, 34398, 35533, 36221, 36985, 39938, 42068, 43742], [32845, 33559, 34488, 35525, 36062, 36905, 39556, 41541, 42931]]
    classwise_dist = {
        CLASSES[0]: [[0.6771, 0.6368, 0.7028, 0.7445, 0.7622, 0.7125, 0.7993, 0.8932, 0.8544], [0.6755, 0.6567, 0.6839, 0.7429, 0.7557, 0.7087, 0.7724, 0.8617, 0.8479]], 
        CLASSES[1]: [[0.6565, 0.7151, 0.6990, 0.6795, 0.7660, 0.7888, 0.8451, 0.9421, 0.8868], [0.6543, 0.7081, 0.7058, 0.6832, 0.7662, 0.7841, 0.8253, 0.9362, 0.8608]],
        CLASSES[2]: [[0.6633, 0.6653, 0.6704, 0.7069, 0.7012, 0.7997, 0.7564, 0.8617, 0.8029], [0.6372, 0.6400, 0.6565, 0.6969, 0.6747, 0.7979, 0.7541, 0.8441, 0.7470]],
        CLASSES[3]: [[0.6840, 0.6758, 0.7176, 0.7347, 0.6902, 0.7177, 0.8122, 0.8067, 0.9227], [0.6881, 0.6784, 0.7343, 0.7497, 0.6795, 0.7349, 0.7906, 0.8077, 0.8775]],
        CLASSES[4]: [[0.6632, 0.6812, 0.6622, 0.7263, 0.6995, 0.7464, 0.8870, 0.8111, 0.9633], [0.6580, 0.6786, 0.6799, 0.7122, 0.7142, 0.7378, 0.8833, 0.8129, 0.9665]],
        CLASSES[5]: [[0.6709, 0.6963, 0.6908, 0.6706, 0.7400, 0.7361, 0.7875, 0.8349, 0.8811], [0.6629, 0.7004, 0.6744, 0.6553, 0.7330, 0.7280, 0.8078, 0.8132, 0.8654]],
        CLASSES[6]: [[0.6141, 0.6465, 0.6640, 0.6967, 0.7578, 0.7733, 0.7909, 0.9225, 0.8374], [0.6216, 0.6599, 0.6893, 0.6865, 0.7692, 0.7834, 0.7857, 0.9278, 0.8208]],
        CLASSES[7]: [[0.6397, 0.6810, 0.6861, 0.6871, 0.7101, 0.7095, 0.7395, 0.7710, 0.8779], [0.6461, 0.6680, 0.6867, 0.7025, 0.7037, 0.7200, 0.7294, 0.7424, 0.9007]],
        CLASSES[8]: [[0.6626, 0.6791, 0.7200, 0.6959, 0.7007, 0.7316, 0.8600, 0.7309, 0.7522], [0.6654, 0.6716, 0.7192, 0.7109, 0.7185, 0.7223, 0.8733, 0.7084, 0.7331]],
        CLASSES[9]: [[0.6382, 0.6583, 0.6672, 0.7647, 0.7187, 0.6844, 0.7125, 0.8432, 0.9716], [0.6599, 0.6516, 0.6682, 0.7658, 0.7005, 0.6675, 0.6923, 0.8585, 0.9686]]
    }
    plot_guidance_classwise(classwise_dist, color, 0.7)

def call_guidance_dist():
    # 95C-5G
    color = [[47686, 47898, 48198, 48400, 48506, 48697, 49191, 49472, 49643], [47333, 47624, 47947, 48137, 48297, 48423, 49046, 49335, 49572]]
    plot_guidance_dist(color, 0.95)

    # 90C-10G
    color = [[44009, 44496, 45032, 45538, 45794, 46133, 47243, 47982, 48461], [44183, 44713, 45270, 45793, 45943, 46450, 47499, 48229, 48527]]
    # plot_guidance_dist(color, 0.9)

    # 70C-30G
    color = [[32848, 33675, 34398, 35533, 36221, 36985, 39938, 42068, 43742], [32845, 33559, 34488, 35525, 36062, 36905, 39556, 41541, 42931]]
    value_70 = [[9.61, 9.93, 10.36, 10.83, 11.17, 11.65, 13.51, 15.06, 16.70], [9.58, 9.87, 10.30, 10.76, 11.10, 11.53, 13.20, 14.52, 15.80]]
    # plot_guidance_dist(color, 0.7)

    # 50C-50G
    color = [[22076, 22636, 23582, 24674, 25462, 26271, 29939, 33002, 35650], [22535, 23023, 23775, 24658, 25395, 26157, 29422, 32394, 34909]]
    value_50 = [[6.67, 6.90, 7.27, 7.64, 7.97, 8.38, 10.07, 11.52, 13.01], [6.56, 6.78, 7.17, 7.55, 7.86, 8.25, 9.93, 11.35, 12.88]]
    # plot_guidance_dist(color, 0.5)

    # 30C-70G
    color = [[12002, 12399, 13225, 13867, 14553, 15075, 18263, 20895, 23620], [12004, 12442, 12942, 13693, 14082, 14618, 17131, 19095, 21139]]
    value_30 = [[3.67, 3.82, 4.09, 4.32, 4.51, 4.78, 5.94, 6.91, 8.09], [3.58, 3.73, 3.96, 4.19, 4.31, 4.56, 5.50, 6.24, 7.11]]
    # plot_guidance_dist(color, 0.3)

    # 10C-90G
    color = [[3292, 3406, 3731, 3869, 4099, 4364, 5572, 6640, 8276], [3304, 3391, 3738, 3841, 3996, 4262, 5400, 6293, 7539]]
    # plot_guidance_dist(color, 0.1)

    # 5C-95G
    color = [[1255, 1295, 1451, 1522, 1547, 1655, 2156, 2605, 3367], [1310, 1295, 1431, 1456, 1480, 1558, 1921, 2225, 2702]]
    plot_guidance_dist(color, 0.05)


def plot_cifar_colorgray():
    colors = ["blue", "orange", "green", "purple", "brown"]
    domain_split = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float64)
    ws = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    counts = [
        np.vstack([[3292, 3406, 3731, 3869, 4099, 4364, 5572, 6640, 8276], [3304, 3391, 3738, 3841, 3996, 4262, 5400, 6293, 7539]]) / 50000,
        np.vstack([[12002, 12399, 13225, 13867, 14553, 15075, 18263, 20895, 23620], [12004, 12442, 12942, 13693, 14082, 14618, 17131, 19095, 21139]]) / 50000,
        np.vstack([[22076, 22636, 23582, 24674, 25462, 26271, 29939, 33002, 35650], [22535, 23023, 23775, 24658, 25395, 26157, 29422, 32394, 34909]]) / 50000,
        np.vstack([[32848, 33675, 34398, 35533, 36221, 36985, 39938, 42068, 43742], [32845, 33559, 34488, 35525, 36062, 36905, 39556, 41541, 42931]]) / 50000,
        np.vstack([[44009, 44496, 45032, 45538, 45794, 46133, 47243, 47982, 48461], [44183, 44713, 45270, 45793, 45943, 46450, 47499, 48229, 48527]]) / 50000,
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
        plt.title("Sampling distribution w.r.t Guidance on CIFAR10 dataset")
    
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-03-27/cifar10/figures", "guidance_count_colorgray.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-03-27/cifar10/figures", "guidance_count_colorgray.pdf"), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # call_guidance_dist()
    # call_guidance_dist_classwise()
    plot_cifar_colorgray()