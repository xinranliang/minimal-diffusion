import numpy as np 
import os 
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

num_bins = 50
guidance_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

def plot_cifar_imgnet_hist(real_pred_probs, syn_pred_probs, save_folder):
    assert len(guidance_values) == len(syn_pred_probs), "Guidance values do not match"

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        # plt.hist(real_pred_probs, bins=num_bins, histtype="step", label="real samples")
        for idx in range(len(guidance_values)):
            plt.hist(syn_pred_probs[idx], bins=num_bins, histtype="step", label=f"$w = {guidance_values[idx]}$")
        plt.xticks(np.arange(0, 1.01, 0.1), np.arange(0, 1.01, 0.1))
        plt.xlabel("Probability value")
        plt.ylabel("Frequency")
        plt.title("Empirical histogram of predicted probability as from CIFAR domain")
        plt.legend(loc="upper center")
    
    plt.savefig(os.path.join(save_folder, "pred_prob_hist.png"), dpi=300, bbox_inches="tight")

def plot_cifar_imgnet():
    colors = ["blue", "orange", "green", "purple", "brown"]
    domain_split = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float64)
    ws = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)

    counts = [
        np.vstack([[0.11872, 0.1353, 0.13504, 0.12312, 0.10918, 0.09716, 0.08302, 0.07374, 0.06334, 0.0596, 0.05524], [0.11, 0.13556, 0.14092, 0.12522, 0.10672, 0.09176, 0.0733, 0.06334, 0.05338, 0.04602, 0.03978]]),
        np.vstack([[0.37244, 0.44306, 0.45414, 0.43128, 0.39142, 0.34686, 0.30642, 0.26212, 0.22906, 0.19812, 0.16954], [0.34862, 0.42456, 0.431, 0.411, 0.3689, 0.33212, 0.29364, 0.25682, 0.22904, 0.20098, 0.17754]]),
        np.vstack([[0.57248, 0.66018, 0.66522, 0.63276, 0.58502, 0.53106, 0.4729, 0.41948, 0.3697, 0.32244, 0.2854], [0.56128, 0.64854, 0.66134, 0.64172, 0.60918, 0.5732, 0.53122, 0.48484, 0.4507, 0.4084, 0.36716]]),
        np.vstack([[0.75576, 0.8234, 0.82056, 0.78588, 0.73236, 0.65704, 0.58582, 0.51348, 0.45046, 0.39204, 0.3449], [0.7355, 0.8012, 0.7937, 0.75584, 0.70562, 0.63592, 0.56986, 0.50442, 0.44678, 0.3876, 0.3373]]),
        np.vstack([[0.93114, 0.94312, 0.92266, 0.88398, 0.82264, 0.75134, 0.6566, 0.56886, 0.49492, 0.41414, 0.35326], [0.9262, 0.94138, 0.92608, 0.8917, 0.83558, 0.75662, 0.6757, 0.58994, 0.50796, 0.43614, 0.37514]]),
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
        plt.title("Sampling distribution of classified as CIFAR domain w.r.t Guidance")
        # plt.legend()
    
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-31/cifar-imagenet/figures", "guidance_count.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-31/cifar-imagenet/figures", "guidance_count.pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def compute_sub_error():
    data_dict = {
        0.0: np.array([[0.1169, 0.1171, 0.1185, 0.1181, 0.1208, 0.1051, 0.1121, 0.1081, 0.1078, 0.1102], [0.3792, 0.3718, 0.371, 0.3712, 0.3693, 0.3503, 0.3418, 0.3446, 0.358, 0.3544], [0.5817, 0.5743, 0.5732, 0.5673, 0.5709, 0.5707, 0.5576, 0.5629, 0.5599, 0.5623], [0.7491, 0.7512, 0.7584, 0.7584, 0.7573, 0.7345, 0.7429, 0.7323, 0.7354, 0.7313], [0.9344, 0.9305, 0.934, 0.9333, 0.9243, 0.9282, 0.9231, 0.9294, 0.926, 0.928]]),
        1.0: np.array([[0.1345, 0.1325, 0.1323, 0.1349, 0.1352, 0.1402, 0.1345, 0.1338, 0.1326, 0.1357], [0.4452, 0.4453, 0.444, 0.4422, 0.4449, 0.422, 0.4179, 0.4183, 0.424, 0.4194], [0.6578, 0.6577, 0.6638, 0.6616, 0.6553, 0.6514, 0.6428, 0.6465, 0.6547, 0.6586], [0.818, 0.8187, 0.8235, 0.8235, 0.8237, 0.8013, 0.8037, 0.8018, 0.8012, 0.7963], [0.9442, 0.9425, 0.9426, 0.9447, 0.9462, 0.9397, 0.9414, 0.9433, 0.9455, 0.9391]]),
        2.0: np.array([[0.1345, 0.1318, 0.1345, 0.1386, 0.1354, 0.1353, 0.1452, 0.1424, 0.1406, 0.1406], [0.4591, 0.4523, 0.455, 0.4502, 0.4495, 0.434, 0.4292, 0.4311, 0.4331, 0.4364], [0.6591, 0.6678, 0.6583, 0.6639, 0.6617, 0.6613, 0.6574, 0.6583, 0.6697, 0.6597], [0.8177, 0.8197, 0.8159, 0.8216, 0.8238, 0.7914, 0.7956, 0.7976, 0.7923, 0.7934], [0.9211, 0.9278, 0.9247, 0.9217, 0.9195, 0.9287, 0.9254, 0.9247, 0.9283, 0.9276]]),
        3.0: np.array([[0.1155, 0.1205, 0.1206, 0.1265, 0.1244, 0.122, 0.1228, 0.1272, 0.1243, 0.1235], [0.4267, 0.4283, 0.443, 0.4278, 0.4297, 0.4039, 0.4034, 0.4154, 0.4128, 0.415], [0.6315, 0.6315, 0.6358, 0.6356, 0.6319, 0.6386, 0.6374, 0.6399, 0.6508, 0.6374], [0.787, 0.7873, 0.7819, 0.791, 0.7837, 0.7593, 0.7558, 0.7556, 0.7558, 0.7549], [0.883, 0.8888, 0.883, 0.8801, 0.8828, 0.8947, 0.8908, 0.8946, 0.8913, 0.8923]]),
        4.0: np.array([[0.1075, 0.1069, 0.1153, 0.1064, 0.1073, 0.1043, 0.1072, 0.1042, 0.1091, 0.1072], [0.3906, 0.3927, 0.389, 0.3903, 0.3972, 0.3727, 0.3651, 0.3745, 0.3648, 0.3618], [0.5916, 0.5821, 0.5805, 0.5794, 0.5915, 0.6061, 0.608, 0.605, 0.6105, 0.605], [0.7235, 0.7375, 0.7341, 0.7316, 0.7393, 0.7035, 0.7009, 0.7032, 0.6985, 0.7077], [0.8318, 0.8201, 0.827, 0.8223, 0.8211, 0.8361, 0.8292, 0.8364, 0.8324, 0.834]]),
        5.0: np.array([[0.0962, 0.0973, 0.0978, 0.0988, 0.0959, 0.0943, 0.0934, 0.092, 0.0913, 0.0923], [0.3457, 0.3452, 0.3489, 0.354, 0.3408, 0.3359, 0.336, 0.3311, 0.335, 0.3238], [0.5303, 0.5317, 0.5352, 0.5311, 0.5323, 0.5721, 0.5697, 0.5786, 0.5714, 0.572], [0.6603, 0.6455, 0.6598, 0.6546, 0.6628, 0.6451, 0.6337, 0.638, 0.6419, 0.642], [0.7485, 0.7492, 0.7505, 0.755, 0.7575, 0.7556, 0.759, 0.7555, 0.7547, 0.7577]]),
        6.0: np.array([[0.0798, 0.0825, 0.0835, 0.0799, 0.0814, 0.0761, 0.0722, 0.0734, 0.0728, 0.0736], [0.2978, 0.3047, 0.3021, 0.3038, 0.304, 0.2878, 0.285, 0.2909, 0.2952, 0.2935], [0.469, 0.477, 0.4749, 0.467, 0.4651, 0.5305, 0.5298, 0.5387, 0.5322, 0.5326], [0.5819, 0.5806, 0.5847, 0.5941, 0.5816, 0.579, 0.5764, 0.5637, 0.5671, 0.5711], [0.6637, 0.6658, 0.6567, 0.6532, 0.6592, 0.6775, 0.6817, 0.6737, 0.6711, 0.6805]]),
        7.0: np.array([[0.0739, 0.0789, 0.0724, 0.0735, 0.0765, 0.0636, 0.066, 0.0625, 0.0643, 0.0626], [0.2606, 0.2557, 0.2554, 0.2613, 0.2579, 0.2515, 0.2565, 0.2563, 0.2634, 0.2601], [0.4223, 0.4175, 0.4253, 0.4186, 0.4148, 0.4866, 0.4891, 0.4817, 0.485, 0.4851], [0.5152, 0.5158, 0.5172, 0.5203, 0.515, 0.5041, 0.4983, 0.5037, 0.5009, 0.5063], [0.5661, 0.569, 0.5711, 0.5629, 0.57, 0.5866, 0.5897, 0.6008, 0.5941, 0.5851]]),
        8.0: np.array([[0.0606, 0.065, 0.0646, 0.0641, 0.0625, 0.054, 0.049, 0.0551, 0.0556, 0.0531], [0.2263, 0.2306, 0.2218, 0.2264, 0.223, 0.2299, 0.2331, 0.2312, 0.2278, 0.2307], [0.3694, 0.3738, 0.3712, 0.3698, 0.367, 0.457, 0.4489, 0.4567, 0.4468, 0.4521], [0.4461, 0.4566, 0.4449, 0.456, 0.4551, 0.4479, 0.4485, 0.448, 0.4542, 0.4474], [0.4919, 0.492, 0.4936, 0.4891, 0.4934, 0.504, 0.501, 0.5086, 0.5183, 0.511]]),
        9.0: np.array([[0.0574, 0.0632, 0.0586, 0.0626, 0.0564, 0.0458, 0.0461, 0.0474, 0.0464, 0.0477], [0.201, 0.1961, 0.1981, 0.1989, 0.1974, 0.1995, 0.1976, 0.2044, 0.2011, 0.1998], [0.3222, 0.3191, 0.3168, 0.3192, 0.3159, 0.4044, 0.402, 0.4074, 0.4013, 0.4074], [0.3872, 0.3907, 0.3925, 0.3849, 0.3931, 0.3952, 0.3867, 0.3911, 0.3903, 0.3874], [0.4104, 0.4143, 0.4158, 0.4135, 0.4194, 0.4398, 0.442, 0.4334, 0.437, 0.4311]]),
        10.0: np.array([[0.0562, 0.0549, 0.0557, 0.0588, 0.055, 0.0413, 0.0398, 0.0401, 0.0372, 0.04], [0.1675, 0.1728, 0.1658, 0.174, 0.1746, 0.1763, 0.1724, 0.1822, 0.1714, 0.1781], [0.2809, 0.2872, 0.287, 0.2863, 0.281, 0.3634, 0.3635, 0.3656, 0.3598, 0.3632], [0.3408, 0.3412, 0.3444, 0.3413, 0.3498, 0.3411, 0.3368, 0.3394, 0.338, 0.3418], [0.3559, 0.3516, 0.348, 0.3534, 0.3457, 0.3735, 0.3753, 0.378, 0.3717, 0.3795]]),
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
        plt.title("Sampling distribution of classified as CIFAR domain w.r.t Guidance")
    
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-31/cifar-imagenet/figures", "guidance_count_subset.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-31/cifar-imagenet/figures", "guidance_count_subset.pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_check_domaineval(return_results, save_folder, cfg_w, cf_title):
    # confusion matrix between predicted labels and true labels
    cf_matrix = confusion_matrix(return_results["full"]["true_labels"], return_results["full"]["pred_labels"])
    cf_display = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=["cifar", "imgnet"])
    cf_display.plot()
    plt.title(f"cifar/imgnet = {cf_title[0]}/{cf_title[1]} with cfg_w = {cfg_w}")
    plt.savefig(f"{save_folder}/cf_matrix_cfg_w{cfg_w}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_folder}/cf_matrix_cfg_w{cfg_w}.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"cfg_w value: {cfg_w}")

    # accuracy
    print("accuracy: {:.5f}".format(return_results["full"]["accuracy"]))

    # at representation level, percentage of predicted cifar and true cifar
    print("true portion of cifar samples: {:.5f}".format(return_results["true"]["num_cifar"] / 2500))
    print("predicted portion of cifar samples: {:.5f}".format(return_results["pred"]["num_cifar"] / 2500))

def plot_check_repr():
    ws = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)
    # at representation level, percentage of predicted cifar and true cifar
    true_repr = np.array([0.5112, 0.5032, 0.504, 0.492, 0.4848, 0.5056, 0.492, 0.52, 0.504, 0.4852, 0.524], dtype=np.float64)
    line_labels = ["cifar/imgnet = 0.1/0.9", "cifar/imgnet = 0.5/0.5", "cifar/imgnet = 0.9/0.1"]
    pred_repr = [
        np.array([0.502, 0.502, 0.5028, 0.4912, 0.4848, 0.5052, 0.492, 0.5196, 0.504, 0.4852, 0.5236], dtype=np.float64), # 0.1/0.9
        np.array([0.5028, 0.5004, 0.5012, 0.4876, 0.4816, 0.4992, 0.4892, 0.5156, 0.5016, 0.4804, 0.522], dtype=np.float64), # 0.5/0.5
        np.array([0.5092, 0.4916, 0.4964, 0.4828, 0.4664, 0.4844, 0.4664, 0.4844, 0.4636, 0.444, 0.4684], dtype=np.float64) # 0.9/0.1
    ]

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        plt.plot(ws, true_repr, linewidth=2, linestyle="--", label="True empirical representation")
        for idx in range(len(line_labels)):
            plt.plot(ws, pred_repr[idx], linewidth=1, linestyle="-", label=line_labels[idx])
        plt.xticks(ws, ws)
        plt.ylim(0.44, 0.55)
        plt.yticks(np.arange(0.44, 0.55, step=0.01))
        plt.xlabel("Scale of classifier-free guidance ($w$)")
        plt.ylabel("Predicted empirical representation")
        plt.title("Performance of automatic classifier on CIFAR/ImageNet w.r.t Guidance")
        plt.legend()
    
    plt.savefig("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-31/cifar-imagenet-check/figures/acc_repr_level.png", dpi=300, bbox_inches="tight")
    plt.savefig("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-31/cifar-imagenet-check/figures/acc_repr_level.pdf", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    # os.makedirs("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-31/cifar-imagenet/figures", exist_ok=True)
    # plot_cifar_imgnet()
    # compute_sub_error()
    os.makedirs("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-31/cifar-imagenet-check/figures", exist_ok=True)
    plot_check_repr()