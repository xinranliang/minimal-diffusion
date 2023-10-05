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

def plot_cifar_imgnet(caliberate, correct):
    colors = ["blue", "orange", "green", "purple", "brown"]
    domain_split = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float64)
    ws = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)

    if caliberate == "none":
        counts = [
            np.vstack([[0.119, 0.135, 0.135, 0.123, 0.109, 0.097, 0.083, 0.074, 0.063, 0.060, 0.055], [0.110, 0.136, 0.141, 0.125, 0.107, 0.092, 0.073, 0.063, 0.053, 0.046, 0.040]]),
            np.vstack([[0.372, 0.443, 0.454, 0.431, 0.391, 0.347, 0.306, 0.262, 0.229, 0.198, 0.170], [0.349, 0.425, 0.431, 0.411, 0.369, 0.332, 0.294, 0.257, 0.229, 0.201, 0.178]]),
            np.vstack([[0.572, 0.660, 0.665, 0.633, 0.585, 0.531, 0.473, 0.419, 0.370, 0.322, 0.285], [0.561, 0.649, 0.661, 0.642, 0.609, 0.573, 0.531, 0.485, 0.451, 0.408, 0.367]]),
            np.vstack([[0.756, 0.823, 0.821, 0.786, 0.732, 0.657, 0.586, 0.513, 0.450, 0.392, 0.345], [0.736, 0.801, 0.794, 0.756, 0.706, 0.636, 0.570, 0.504, 0.447, 0.388, 0.337]]),
            np.vstack([[0.931, 0.943, 0.923, 0.884, 0.823, 0.751, 0.657, 0.569, 0.495, 0.414, 0.353], [0.926, 0.941, 0.926, 0.892, 0.836, 0.757, 0.676, 0.590, 0.508, 0.436, 0.375]]),
        ]
    elif caliberate == "reweight":
        counts = [
            np.vstack([[0.133, 0.150, 0.151, 0.137, 0.122, 0.108, 0.093, 0.082, 0.071, 0.066, 0.062], [0.123, 0.150, 0.155, 0.138, 0.118, 0.101, 0.082, 0.070, 0.060, 0.052, 0.044]]),
            np.vstack([[0.384, 0.456, 0.466, 0.445, 0.405, 0.359, 0.319, 0.277, 0.243, 0.211, 0.183], [0.361, 0.438, 0.444, 0.425, 0.383, 0.347, 0.308, 0.270, 0.244, 0.215, 0.191]]),
            np.vstack([[0.585, 0.673, 0.681, 0.648, 0.603, 0.550, 0.495, 0.443, 0.394, 0.349, 0.311], [0.574, 0.661, 0.677, 0.658, 0.626, 0.593, 0.554, 0.508, 0.478, 0.436, 0.399]]),
            np.vstack([[0.769, 0.838, 0.837, 0.805, 0.754, 0.682, 0.613, 0.541, 0.478, 0.420, 0.372], [0.750, 0.815, 0.810, 0.774, 0.725, 0.660, 0.597, 0.534, 0.477, 0.420, 0.370]]),
            np.vstack([[0.943, 0.956, 0.939, 0.905, 0.849, 0.782, 0.693, 0.609, 0.534, 0.456, 0.395], [0.938, 0.955, 0.944, 0.915, 0.869, 0.799, 0.725, 0.645, 0.565, 0.495, 0.435]]),
        ]
    
    if correct:
        if caliberate == "none":
            coeff = [
                np.array([1.019, 1.003, 1.002, 1.001, 0.999, 1.000, 1.000, 1.000, 1.001, 1.000, 1.000], dtype=float),
                np.array([1.014, 1.004, 1.005, 1.005, 1.004, 1.004, 1.004, 1.003, 1.004, 1.005, 1.002], dtype=float),
                np.array([1.008, 1.005, 1.008, 1.008, 1.009, 1.007, 1.007, 1.006, 1.008, 1.010, 1.004], dtype=float),
                np.array([1.008, 1.013, 1.013, 1.017, 1.022, 1.029, 1.029, 1.044, 1.049, 1.064, 1.069], dtype=float),
                np.array([1.008, 1.020, 1.018, 1.025, 1.034, 1.051, 1.050, 1.081, 1.091, 1.119, 1.134], dtype=float),
            ]
        elif caliberate == "reweight":
            coeff = [
                np.array([0.994, 0.998, 0.996, 0.998, 0.997, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000], dtype=float),
                np.array([0.992, 0.999, 1.000, 1.000, 1.001, 1.001, 1.001, 1.001, 1.001, 1.001, 1.001], dtype=float),
                np.array([0.991, 1.000, 1.004, 1.002, 1.004, 1.003, 1.002, 1.002, 1.002, 1.003, 1.002], dtype=float),
                np.array([0.990, 1.003, 1.006, 1.005, 1.009, 1.013, 1.013, 1.023, 1.027, 1.040, 1.041], dtype=float),
                np.array([0.988, 1.006, 1.008, 1.008, 1.013, 1.023, 1.024, 1.045, 1.051, 1.077, 1.080], dtype=float),
            ]
        
        # multiply by correction coefficient
        for idx in range(len(domain_split)):
            print("before correction: ", counts[idx])
            counts[idx] = counts[idx] * coeff[idx]
            print("after correction: ", counts[idx])
            input("press enter to continue")

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
        plt.ylabel("Portion classified as CIFAR domain")
        plt.title(f"Sampling distribution on CIFAR/ImageNet w/ caliberation = {caliberate} and correction = {correct}", fontsize=10)
        # plt.legend()
    
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-31/cifar-imagenet/figures", f"guidance_autocount_caliberate_{caliberate}_correct_{correct}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-31/cifar-imagenet/figures", f"guidance_autocount_caliberate_{caliberate}_correct_{correct}.pdf"), dpi=300, bbox_inches="tight")
    plt.close()



def plot_check_domaineval(return_results, save_folder, cfg_w, cf_title, caliberate):
    # confusion matrix between predicted labels and true labels
    cf_matrix = confusion_matrix(return_results["full"]["true_labels"], return_results["full"]["pred_labels"])
    cf_display = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=["cifar", "imgnet"])
    cf_display.plot()
    plt.title(f"cifar/imgnet = {cf_title[0]}/{cf_title[1]} and cfg_w = {cfg_w} w/ posthoc = {caliberate}")
    plt.savefig(f"{save_folder}/cf_matrix_cfg_w{cfg_w}_posthoc_{caliberate}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_folder}/cf_matrix_cfg_w{cfg_w}_posthoc_{caliberate}.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"cfg_w value: {cfg_w}")

    # accuracy
    print("accuracy: {:.5f}".format(return_results["full"]["accuracy"]))
    # at representation level, percentage of predicted cifar and true cifar
    assert return_results["true"]["num_cifar"] == 2500, "cifar and imgnet should have same number of samples"
    print("predicted portion of cifar samples: {:.5f} with caliberation {}".format(return_results["pred"]["num_cifar"] / 5000, caliberate))

def plot_check_repr(caliberate):
    ws = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)
    # at representation level, percentage of predicted cifar and true cifar
    true_repr = np.array([0.5] * len(ws), dtype=np.float64)
    line_labels = ["cifar/imgnet = 0.1/0.9", "cifar/imgnet = 0.5/0.5", "cifar/imgnet = 0.9/0.1"]

    if caliberate == "none":
        pred_repr = [
            np.array([0.4906, 0.4984, 0.4992, 0.4996, 0.5006, 0.5, 0.4998, 0.5, 0.4994, 0.5002, 0.5], dtype=np.float64), # 0.1/0.9
            np.array([0.496, 0.4974, 0.4962, 0.4958, 0.4954, 0.4964, 0.4964, 0.4968, 0.4962, 0.4952, 0.4982], dtype=np.float64), # 0.5/0.5
            np.array([0.4958, 0.49, 0.4912, 0.488, 0.4836, 0.4758, 0.4762, 0.4624, 0.4582, 0.4468, 0.441], dtype=np.float64) # 0.9/0.1
        ]
    elif caliberate == "reweight":
        pred_repr = [
            np.array([0.5032, 0.501, 0.502, 0.5008, 0.5016, 0.5, 0.5002, 0.5, 0.5, 0.5002, 0.5], dtype=np.float64), # 0.1/0.9
            np.array([0.5044, 0.5, 0.498, 0.4992, 0.4978, 0.4986, 0.499, 0.499, 0.499, 0.4984, 0.4988], dtype=np.float64), # 0.5/0.5
            np.array([0.506, 0.4968, 0.4962, 0.4962, 0.4934, 0.4888, 0.4882, 0.4786, 0.4756, 0.4644, 0.463], dtype=np.float64) # 0.9/0.1
        ]

    plt.figure(figsize=(8, 8/1.6))
    with plt.style.context('ggplot'):
        plt.plot(ws, true_repr, linewidth=3, linestyle="--", label="True empirical representation")
        for idx in range(len(line_labels)):
            plt.plot(ws, pred_repr[idx], linewidth=2, linestyle="-", label=line_labels[idx])
        plt.xticks(ws, ws)
        plt.ylim(0.43, 0.51)
        plt.yticks(np.arange(0.43, 0.51, step=0.01))
        plt.xlabel("Scale of classifier-free guidance ($w$)")
        plt.ylabel("Predicted empirical representation")
        plt.title(f"Automatic classifier w/ caliberate = {caliberate} on CIFAR/ImageNet w.r.t cfg_w")
        plt.legend()
    
    plt.savefig(f"/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-31/cifar-imagenet-check/figures/repr_acc_posthoc_{caliberate}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-31/cifar-imagenet-check/figures/repr_acc_posthoc_{caliberate}.pdf", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    # report results
    os.makedirs("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-31/cifar-imagenet/figures", exist_ok=True)
    plot_cifar_imgnet(caliberate="none", correct=True)
    plot_cifar_imgnet(caliberate="reweight", correct=True)
    plot_cifar_imgnet(caliberate="none", correct=False)
    plot_cifar_imgnet(caliberate="reweight", correct=False)

    """# check reported results
    os.makedirs("/n/fs/xl-diffbia/projects/minimal-diffusion/logs/2023-07-31/cifar-imagenet-check/figures", exist_ok=True)
    plot_check_repr(caliberate="none")
    plot_check_repr(caliberate="reweight")"""