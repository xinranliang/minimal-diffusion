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


def plot_fairface():
    colors = ["blue", "orange", "green", "purple", "brown"]
    domain_split = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float64)
    ws = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)

    counts = [
        np.vstack([[0.14264, 0.12334, 0.11258, 0.10372, 0.0985, 0.09578, 0.09238, 0.09126, 0.08698, 0.087, 0.08724], [0.14734, 0.14028, 0.13536, 0.1319, 0.13594, 0.1319, 0.13174, 0.13048, 0.12368, 0.12362, 0.12218]]),
        np.vstack([[0.3326, 0.32388, 0.31862, 0.3092, 0.30806, 0.30466, 0.30514, 0.2991, 0.29528, 0.29108, 0.29252], [0.3293, 0.31766, 0.30828, 0.2919, 0.28778, 0.27792, 0.27094, 0.26382, 0.2585, 0.25608, 0.25804]]),
        np.vstack([[0.51302, 0.50428, 0.49352, 0.46838, 0.44752, 0.42246, 0.4028, 0.38368, 0.36376, 0.35416, 0.34942], [0.51802, 0.5238, 0.5313, 0.52344, 0.51474, 0.50152, 0.4902, 0.47748, 0.47126, 0.4636, 0.45234]]),
        np.vstack([[0.69242, 0.6948, 0.69158, 0.66634, 0.6496, 0.62544, 0.59754, 0.58262, 0.56452, 0.55334, 0.54616], [0.69038, 0.70468, 0.70588, 0.70744, 0.70056, 0.68948, 0.6775, 0.6682, 0.65454, 0.65248, 0.6457]]),
        np.vstack([[0.86654, 0.87968, 0.8842, 0.87756, 0.87022, 0.85892, 0.84194, 0.83222, 0.81674, 0.80674, 0.79784], [0.8698, 0.88768, 0.90002, 0.90224, 0.89954, 0.89232, 0.88674, 0.8747, 0.86382, 0.85678, 0.84932]]),
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
        plt.title("Sampling distribution over 50k of classified as Female w.r.t Guidance on FairFace")
    
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-25/fairface/figures", "guidance_count_gender.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-25/fairface/figures", "guidance_count_gender.pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def compute_sub_error():
    data_dict = {
        0.0: np.array([[0.1429, 0.1413, 0.1474, 0.1449, 0.1442, 0.1447, 0.1447, 0.1463, 0.1458, 0.1487], [0.3331, 0.3328, 0.3355, 0.3336, 0.3236, 0.3323, 0.3263, 0.3253, 0.3285, 0.3349], [0.509, 0.5115, 0.5162, 0.5135, 0.5116, 0.5237, 0.5245, 0.5169, 0.5122, 0.5173], [0.6965, 0.6939, 0.6948, 0.6949, 0.6921, 0.6971, 0.6856, 0.6848, 0.6927, 0.6971], [0.8632, 0.8654, 0.8722, 0.868, 0.8657, 0.8652, 0.8689, 0.8686, 0.8682, 0.8677]]),
        1.0: np.array([[0.1241, 0.1241, 0.1289, 0.1275, 0.1239, 0.1422, 0.1429, 0.135, 0.1371, 0.1391], [0.3218, 0.3175, 0.3275, 0.3222, 0.3278, 0.3225, 0.3242, 0.3202, 0.3129, 0.3247], [0.5088, 0.51, 0.505, 0.5116, 0.4999, 0.5241, 0.5231, 0.5208, 0.53, 0.5189], [0.6999, 0.6868, 0.6982, 0.6873, 0.6945, 0.7059, 0.7066, 0.7027, 0.7025, 0.7043], [0.8843, 0.8748, 0.8831, 0.877, 0.88, 0.8897, 0.8904, 0.8876, 0.8908, 0.888]]),
        2.0: np.array([[0.1147, 0.114, 0.1142, 0.1103, 0.1111, 0.1346, 0.1285, 0.1364, 0.1333, 0.1322], [0.3233, 0.3256, 0.3152, 0.3162, 0.318, 0.311, 0.3069, 0.305, 0.3068, 0.3098], [0.4865, 0.4972, 0.4976, 0.4956, 0.4896, 0.5316, 0.5409, 0.5359, 0.5336, 0.5328], [0.6886, 0.6902, 0.6911, 0.6972, 0.6906, 0.7076, 0.7037, 0.7028, 0.7094, 0.7032], [0.8801, 0.883, 0.8802, 0.8866, 0.886, 0.9043, 0.9023, 0.9007, 0.9026, 0.9008]]),
        3.0: np.array([[0.1033, 0.1049, 0.1014, 0.103, 0.1077, 0.1364, 0.1297, 0.1267, 0.1336, 0.1315], [0.3096, 0.3115, 0.3038, 0.3105, 0.3075, 0.2946, 0.2934, 0.2934, 0.2905, 0.2949], [0.4691, 0.4729, 0.4723, 0.4755, 0.4706, 0.523, 0.5282, 0.5295, 0.5238, 0.52], [0.6668, 0.6676, 0.6646, 0.6591, 0.6647, 0.7073, 0.7202, 0.7047, 0.7077, 0.7065], [0.8808, 0.8761, 0.8774, 0.8781, 0.8813, 0.8984, 0.9044, 0.9043, 0.9058, 0.9042]]),
        4.0: np.array([[0.0974, 0.1026, 0.095, 0.0954, 0.0989, 0.1376, 0.1333, 0.1368, 0.1357, 0.1321], [0.3066, 0.3133, 0.3072, 0.307, 0.3084, 0.2897, 0.2881, 0.2949, 0.2878, 0.2916], [0.4453, 0.4525, 0.4446, 0.4378, 0.4516, 0.5032, 0.5157, 0.5193, 0.5108, 0.5153], [0.6454, 0.6528, 0.6534, 0.6509, 0.6484, 0.707, 0.6969, 0.6933, 0.7034, 0.7066], [0.8723, 0.8745, 0.8701, 0.8711, 0.8733, 0.8976, 0.9033, 0.896, 0.8986, 0.8964]]),
        5.0: np.array([[0.0963, 0.0933, 0.0993, 0.0948, 0.0925, 0.1244, 0.1352, 0.1326, 0.1337, 0.1325], [0.3143, 0.3091, 0.3062, 0.3075, 0.3029, 0.2792, 0.2719, 0.2799, 0.2776, 0.2819], [0.4198, 0.4175, 0.4195, 0.4228, 0.4156, 0.4988, 0.5009, 0.5001, 0.5048, 0.5095], [0.6179, 0.625, 0.6217, 0.6283, 0.6225, 0.6854, 0.6853, 0.6876, 0.6944, 0.6924], [0.8574, 0.8555, 0.862, 0.8615, 0.8541, 0.8904, 0.8945, 0.8926, 0.8907, 0.8956]]),
        6.0: np.array([[0.0927, 0.0977, 0.0938, 0.0941, 0.0929, 0.1298, 0.135, 0.1297, 0.1293, 0.1313], [0.3041, 0.2969, 0.3032, 0.3037, 0.3016, 0.2665, 0.2733, 0.2705, 0.271, 0.2787], [0.4053, 0.3907, 0.4113, 0.4014, 0.401, 0.4879, 0.4833, 0.4978, 0.4819, 0.4905], [0.6038, 0.6011, 0.5977, 0.6043, 0.5926, 0.6793, 0.6754, 0.6732, 0.681, 0.6823], [0.8379, 0.8432, 0.842, 0.8495, 0.8348, 0.8862, 0.8867, 0.8852, 0.8846, 0.8902]]),
        7.0: np.array([[0.0858, 0.0903, 0.0917, 0.0901, 0.0877, 0.1351, 0.1345, 0.1273, 0.1273, 0.1311], [0.3071, 0.2998, 0.2924, 0.3045, 0.2984, 0.2648, 0.266, 0.262, 0.2702, 0.2662], [0.382, 0.3816, 0.3828, 0.3847, 0.3803, 0.4755, 0.472, 0.4806, 0.4759, 0.4807], [0.5828, 0.5818, 0.587, 0.5856, 0.5847, 0.6672, 0.6673, 0.667, 0.6705, 0.662], [0.831, 0.8304, 0.8261, 0.8302, 0.836, 0.8715, 0.8738, 0.877, 0.8721, 0.8787]]),
        8.0: np.array([[0.0882, 0.0844, 0.087, 0.092, 0.089, 0.1238, 0.123, 0.1194, 0.1239, 0.1233], [0.3039, 0.2996, 0.2861, 0.2893, 0.3002, 0.2567, 0.2573, 0.2562, 0.2577, 0.2572], [0.3628, 0.3608, 0.3646, 0.3696, 0.3641, 0.4667, 0.4759, 0.4719, 0.4687, 0.4716], [0.5621, 0.5628, 0.5603, 0.5688, 0.5708, 0.6541, 0.6556, 0.6522, 0.661, 0.6545], [0.8162, 0.8183, 0.813, 0.8228, 0.8143, 0.8624, 0.8627, 0.8637, 0.8639, 0.8644]]),
        9.0: np.array([[0.0926, 0.0824, 0.0884, 0.0875, 0.0846, 0.1184, 0.124, 0.13, 0.123, 0.1266], [0.2937, 0.2912, 0.2951, 0.2864, 0.2872, 0.2566, 0.2567, 0.2542, 0.2482, 0.2606], [0.3537, 0.3564, 0.3652, 0.3528, 0.3551, 0.4664, 0.4667, 0.4594, 0.4568, 0.4569], [0.5568, 0.5444, 0.5508, 0.5556, 0.5571, 0.6521, 0.6535, 0.6517, 0.6545, 0.6431], [0.8086, 0.8071, 0.8061, 0.8018, 0.8072, 0.8618, 0.8584, 0.8531, 0.8567, 0.8554]]),
        10.0: np.array([[0.0872, 0.083, 0.09, 0.0861, 0.0856, 0.1192, 0.1251, 0.1199, 0.1196, 0.1254], [0.2918, 0.2854, 0.2915, 0.3007, 0.2964, 0.2626, 0.2635, 0.255, 0.2581, 0.2638], [0.3496, 0.347, 0.3516, 0.3466, 0.3531, 0.4487, 0.4507, 0.4519, 0.4478, 0.4536], [0.5533, 0.5505, 0.5424, 0.5516, 0.5411, 0.6441, 0.6466, 0.641, 0.6404, 0.6385], [0.7971, 0.8011, 0.799, 0.8042, 0.8037, 0.8531, 0.8461, 0.8552, 0.8488, 0.8522]]),
    }

    colors = ["blue", "orange", "green", "purple", "brown"]
    domain_split = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float64)
    ws = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)
    summary_dict = {"mean": np.zeros((len(domain_split), len(ws)), dtype=float), "std": np.zeros((len(domain_split), len(ws)), dtype=float)}
    for w_idx in range(len(ws)):
        w_mean = np.mean(data_dict[ws[w_idx]], axis=1)
        w_std = np.std(data_dict[ws[w_idx]], axis=1)
        np.copyto(summary_dict["mean"][:, w_idx], w_mean)
        np.copyto(summary_dict["std"][:, w_idx], w_std)

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        for idx in range(len(domain_split)):
            plt.errorbar(ws, summary_dict["mean"][idx], yerr=summary_dict["std"][idx], linestyle="-", linewidth=1, color=colors[idx], capsize=3)
            plt.axhline(domain_split[idx], xmin=ws[0], xmax=ws[-1], linestyle="--", color=colors[idx])
        plt.xticks(ws, ws)
        plt.ylim(0, 1.01)
        plt.yticks(np.arange(0, 1.01, step=0.1))
        plt.xlabel("Scale of classifier-free guidance ($w$)")
        plt.ylabel("Value")
        plt.title("Sampling distribution over 10k of classified as Female w.r.t Guidance on FairFace")
    
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-25/fairface/figures", "guidance_count_gender_subset.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-25/fairface/figures", "guidance_count_gender_subset.pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_check_repr_batch1():
    ws = np.array([0.0, 5.0, 10.0], dtype=np.float64)
    colors = ["blue", "orange", "green", "purple", "brown"]
    line_labels = ["female/male = 0.1/0.9", "female/male = 0.5/0.5", "female/male = 0.9/0.1"]
    true_repr = [
        np.array([8, 4, 6], dtype=np.float64) / 50, # 0.1/0.9
        np.array([18, 14, 11], dtype=np.float64) / 50, # 0.5/0.5
        np.array([36, 41, 44], dtype=np.float64) / 50 # 0.9/0.1
    ]
    pred_repr = [
        np.array([11, 8, 4], dtype=np.float64) / 50, # 0.1/0.9
        np.array([23, 21, 14], dtype=np.float64) / 50, # 0.5/0.5
        np.array([45, 44, 40], dtype=np.float64) / 50 # 0.9/0.1
    ]

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        for idx in range(len(line_labels)):
            plt.plot(ws, true_repr[idx], linestyle="--", color=colors[2 * idx])
            plt.plot(ws, pred_repr[idx], linestyle="-", color=colors[2 * idx])
        plt.xticks(ws, ws)
        plt.ylim(0, 1.01)
        plt.yticks(np.arange(0, 1.01, step=0.1))
        plt.xlabel("Scale of classifier-free guidance ($w$)")
        plt.ylabel("Empirical predicted v.s. annotated representation")
        plt.title("Performance of automatic classifier on FairFace w.r.t Guidance")
    
    plt.savefig("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-25/fairface/figures/acc_repr_level_batch1.png", dpi=300, bbox_inches="tight")
    plt.savefig("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-25/fairface/figures/acc_repr_level_batch1.pdf", dpi=300, bbox_inches="tight")

def plot_check_repr_batch2():
    ws = np.array([0.0, 5.0, 10.0], dtype=np.float64)
    colors = ["blue", "orange", "green", "purple", "brown"]
    line_labels = ["female/male = 0.1/0.9", "female/male = 0.5/0.5", "female/male = 0.9/0.1"]
    true_repr = [
        np.array([10, 2, 6], dtype=np.float64) / 50, # 0.1/0.9
        np.array([30, 24, 23], dtype=np.float64) / 50, # 0.5/0.5
        np.array([40, 44, 39], dtype=np.float64) / 50 # 0.9/0.1
    ]
    pred_repr = [
        np.array([11, 6, 7], dtype=np.float64) / 50, # 0.1/0.9
        np.array([23, 21, 14], dtype=np.float64) / 50, # 0.5/0.5
        np.array([45, 44, 40], dtype=np.float64) / 50 # 0.9/0.1
    ]

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        for idx in range(len(line_labels)):
            plt.plot(ws, true_repr[idx], linestyle="--", color=colors[2 * idx])
            plt.plot(ws, pred_repr[idx], linestyle="-", color=colors[2 * idx])
        plt.xticks(ws, ws)
        plt.ylim(0, 1.01)
        plt.yticks(np.arange(0, 1.01, step=0.1))
        plt.xlabel("Scale of classifier-free guidance ($w$)")
        plt.ylabel("Empirical predicted v.s. annotated representation")
        plt.title("Performance of automatic classifier on FairFace w.r.t Guidance")
    
    plt.savefig("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-25/fairface/figures/acc_repr_level_batch2.png", dpi=300, bbox_inches="tight")
    plt.savefig("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-25/fairface/figures/acc_repr_level_batch2.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-25/fairface/figures", exist_ok=True)
    # plot_fairface()
    # compute_sub_error()
    plot_check_repr_batch1()
    plot_check_repr_batch2()