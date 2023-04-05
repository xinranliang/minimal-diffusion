import numpy as np 
import collections
import os 
import argparse
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

def natural_color_loader(dataset):
    if dataset == "cifar10":
        cifar10_normalize = datasets.CIFAR10(
            root="/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10",
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
            target_transform=None,
            download=False,
        )
        dataloader_normalize = DataLoader(cifar10_normalize, batch_size=len(cifar10_normalize), shuffle=False)
        natural_normalize_batch = natural_color_channel_std(dataloader_normalize, transform=True)
        plot_natural_std(natural_normalize_batch, "normalize_channel_std_mean", "/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10/color_gray_dist")

        cifar10_pixel = datasets.CIFAR10(
            root="/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10",
            train=True,
            transform=transforms.Compose(
                [
                    transforms.PILToTensor(),
                ]
            ),
            target_transform=None,
            download=False,
        )
        dataloader_pixel = DataLoader(cifar10_pixel, batch_size=len(cifar10_pixel), shuffle=False)
        natural_pixel_batch = natural_color_channel_std(dataloader_pixel)
        plot_natural_std(natural_pixel_batch, "pixel_channel_std_mean", "/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10/color_gray_dist")
    
    elif dataset == "mix-cifar10-imagenet":
        mix_normalize = datasets.ImageFolder(
            root = "/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/train",
            transform=transforms.Compose([transforms.ToTensor(),]),
            target_transform=None
        )
        dataloader_normalize = DataLoader(mix_normalize, batch_size=len(mix_normalize), shuffle=False)
        natural_normalize_batch = natural_color_channel_std(dataloader_normalize, transform=True)
        plot_natural_std(natural_normalize_batch, "normalize_channel_std_mean", "/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/color_gray_dist")

        mix_pixel = datasets.ImageFolder(
            root = "/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/train",
            transform=transforms.Compose([transforms.PILToTensor(),]),
            target_transform=None
        )
        dataloader_pixel = DataLoader(mix_pixel, batch_size=len(mix_pixel), shuffle=False)
        natural_pixel_batch = natural_color_channel_std(dataloader_pixel, transform=True)
        plot_natural_std(natural_pixel_batch, "pixel_channel_std_mean", "/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/color_gray_dist")


def natural_color_channel_std(dataloader, transform=False):
    # only 1 batch
    for idx, batch in enumerate(dataloader):
        image, label = batch
        if transform:
            image = image * 2 - 1.0
        samples = image.view(image.shape[0], image.shape[2], image.shape[3], image.shape[1]).cpu().numpy()
        # print(samples.shape) # num_samples x height x width x n_channel
        channel_std_batch = np.std(samples, axis=-1) # num_samples x height x width
        channel_std_batch = channel_std_batch.reshape((channel_std_batch.shape[0], channel_std_batch.shape[1] * channel_std_batch.shape[2])) # num_samples x (height x width)
        channel_std_batch = np.mean(channel_std_batch, axis=-1) # num_samples x 1
        print(channel_std_batch.shape)
    return channel_std_batch.tolist() # should be an array and conver to list

def plot_natural_std(channel_std, title, log_dir):
    r"""
    input: std over RGB channel (num_samples x 1)
    output: plot distribution of channel std
    """

    num_bins = 50
    values, bins = np.histogram(channel_std, num_bins)
    pdf = values / sum(values)
    cdf = np.cumsum(pdf)
    plt.plot(bins[1:], cdf)

    plt.xlabel('RGB channel std value', fontsize=14)
    plt.ylabel('Empirical CDF', fontsize=14)
    plt.title('Empirical CDF of Natural Color Images', fontsize=14)

    os.makedirs(os.path.join(log_dir, "figures"), exist_ok=True)
    plt.savefig(os.path.join(log_dir, "figures", f"{title}_cdf.png"), dpi=80)
    plt.savefig(os.path.join(log_dir, "figures", f"{title}_cdf.pdf"), dpi=80)
    plt.close()

    values, bins = np.histogram(channel_std, num_bins)
    pdf = values / sum(values)
    plt.plot(bins[1:], pdf)

    plt.xlabel('RGB channel std value', fontsize=14)
    plt.ylabel('Empirical PDF', fontsize=14)
    plt.title('Empirical PDF of Natural Color Images', fontsize=14)

    plt.savefig(os.path.join(log_dir, "figures", f"{title}_pdf.png"), dpi=80)
    plt.savefig(os.path.join(log_dir, "figures", f"{title}_pdf.pdf"), dpi=80)
    plt.close()

    print("Minimum value {} in Figure {}".format(min(channel_std), title))


def count_colorgray(samples, threshold, labels=None):
    # input: numpy array of samples in np.uint8 format
    # optional input: labels of each synthetic sample
    # output: number of color and number of gray
    # check std of channel
    channel_std = np.std(samples, axis=-1) # num_samples x height x width
    channel_std = channel_std.reshape((channel_std.shape[0], channel_std.shape[1] * channel_std.shape[2])) # num_samples x (height x width)
    channel_std = np.mean(channel_std, axis=-1) # num_samples x 1

    if labels is None:
        num_color = np.sum(channel_std >= threshold)
        num_gray = np.sum(channel_std < threshold)
        assert num_color + num_gray == channel_std.shape[0]
        return {"num_color": num_color, "num_gray": num_gray}
    else:
        # output: list of dictionary, one element for each class
        ratio_color_classwise = []
        ratio_gray_classwise = []
        assert samples.shape[0] == labels.shape[0], "number of images does not match between samples and labels!"
        class_counter = collections.Counter(labels)
        for class_index in class_counter.keys():
            sample_index = np.where(labels == class_index)[0]
            channel_std_select = np.take(channel_std, sample_index, axis=0)
            ratio_color = np.mean(channel_std_select >= threshold)
            ratio_gray = np.mean(channel_std_select < threshold)
            ratio_color_classwise.append(ratio_color)
            ratio_gray_classwise.append(ratio_gray)
        return {"ratio_color_classwise": ratio_color_classwise, "ratio_gray_classwise": ratio_gray_classwise}

def compute_colorgray(samples, labels=None):
    # input: numpy array of samples in np.uint8 format
    # optional input: labels of each synthetic sample
    # output: value of metrics
    # check std of channel
    channel_std = np.std(samples, axis=-1) # num_samples x height x width
    channel_std = channel_std.reshape((channel_std.shape[0], channel_std.shape[1] * channel_std.shape[2])) # num_samples x (height x width)
    channel_std = np.mean(channel_std, axis=-1) # num_samples x 1

    if labels is None:
        channel_std_group = np.mean(channel_std)
        return channel_std_group
    else:
        channel_std_classwise = []
        assert samples.shape[0] == labels.shape[0], "number of images does not match between samples and labels!"
        class_counter = collections.Counter(labels)
        for class_index in class_counter.keys():
            sample_index = np.where(labels == class_index)[0]
            channel_std_select = np.take(channel_std, sample_index, axis=0)
            channel_std_select = np.mean(channel_std_select).item()
            channel_std_classwise.append(channel_std_select)
        return channel_std_classwise


def get_args():
    parser = argparse.ArgumentParser("Measure generation samplng distribution compared to training distribution")

    parser.add_argument('--dataset', type=str, help="dataset used for experiment")
    parser.add_argument('--natural', action='store_true', help="evaluate distribution of channel std in natural color images")
    parser.add_argument('--train-color', type=float, required=False, help="portion of training distribution that are colored images")
    parser.add_argument('--train-gray', type=float, required=False, help="portion of training distribution that are gradyscaled images")
    
    parser.add_argument('--date', type=str, required=False, help="experiment date for logging purpose")
    parser.add_argument('--diffusion-config', type=str, required=False, default="UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512", help="diffusion model configuration, currently set to be default parameters")

    parser.add_argument('--plot', action="store_true", help="plot figure and save")
    parser.add_argument('--count', action="store_true", help="count number and print")

    args = parser.parse_args()

    if not args.natural:
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

    if args.natural:
        natural_color_loader(args.dataset)
        return

    channel_std = load_sample(args)
    if args.plot:
        plot_channel_std(channel_std, args.log_dir)
    if args.count:
        count_channel_std(channel_std)

    print("========")


if __name__ == "__main__":
    main()