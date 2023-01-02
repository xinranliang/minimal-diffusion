import numpy as np 
import os 
import argparse
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter


def load_sample(args):
    save_file = np.load(args.sample_file, allow_pickle=True)

    samples = save_file['arr_0'] # shape = num_samples x height x width x n_channel
    labels = save_file['arr_1'] # empty if class_cond = False

    # check std of channel
    channel_std = np.std(samples, axis=-1) # num_samples x height x width
    channel_std = channel_std.reshape((channel_std.shape[0], channel_std.shape[1] * channel_std.shape[2])) # num_samples x (height x width)
    channel_std = np.mean(channel_std, axis=-1) # num_samples x 1

    return channel_std

def plot_channel_std(channel_std, log_dir):
    r"""
    input: std over RGB channel (num_samples x 1)
    output: plot distribution of channel std
    """

    num_bins = channel_std.shape[0] // 500
    n, bins, patches = plt.hist(channel_std, num_bins)
    plt.xlabel('RGB channel std value')
    plt.ylabel('Number of Samples')
    plt.title('Histogram of RGB channel std value')
    plt.show()

    os.makedirs(os.path.join(log_dir, "figures"), exist_ok=True)
    plt.savefig(os.path.join(log_dir, "figures", "channel_std.png"), dpi=80)
    plt.savefig(os.path.join(log_dir, "figures", "channel_std.pdf"), dpi=80)

def count_channel_std(channel_std, threshold=1.0):
    r"""
    input: std over RGB channel (num_samples x 1)
    output: count number of color vs grayscale images specified by channel-wise std
    """
    num_color = np.sum(channel_std >= threshold)
    num_gray = np.sum(channel_std < threshold)

    assert num_color + num_gray == channel_std.shape[0]

    print("Number of colored samples: {}".format(num_color))
    print("Number of grayscaled samples: {}".format(num_gray))


def get_args():
    parser = argparse.ArgumentParser("Measure generation samplng distribution compared to training distribution")

    parser.add_argument('--dataset', type=str, help="dataset used for experiment")
    parser.add_argument('--train-color', type=float, help="portion of training distribution that are colored images")
    parser.add_argument('--train-gray', type=float, help="portion of training distribution that are gradyscaled images")
    
    parser.add_argument('--date', type=str, help="experiment date for logging purpose")
    parser.add_argument('--diffusion-config', type=str, default="UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512", help="diffusion model configuration, currently set to be default parameters")

    parser.add_argument('--plot', action="store_true", help="plot figure and save")
    parser.add_argument('--count', action="store_true", help="count number and print")

    args = parser.parse_args()

    args.log_dir = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs", 
                        args.date, args.dataset, "color{}_gray{}".format(args.train_color, args.train_gray), 
                        args.diffusion_config)

    args.sample_file = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs", 
                        args.date, args.dataset, "color{}_gray{}".format(args.train_color, args.train_gray), 
                        args.diffusion_config, "samples", 
                        "{}_color{}_gray{}_epoch_{}_num{}.npz".format(args.dataset, args.train_color, args.train_gray, 950, 50000))
    
    return args


def main():
    args = get_args()
    print("Evaluating training distribution with {} colored and {} grayscaled".format(args.train_color, args.train_gray))

    channel_std = load_sample(args)
    if args.plot:
        plot_channel_std(channel_std, args.log_dir)
    if args.count:
        count_channel_std(channel_std)

    print("========")


if __name__ == "__main__":
    main()