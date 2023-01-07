import numpy as np 
import os 
import argparse
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def process_real_dataset(dataset_name, dataset_dir="./datasets"):
    dataset_dir = os.path.join(dataset_dir, dataset_name)

    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(
            root=dataset_dir,
            train=True,
            transform=transforms.ToTensor()
        )
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        batch_array, labels = next(iter(loader))

        batch_array = batch_array.numpy()
        labels = labels.numpy()

        np.savez(
            os.path.join(dataset_dir, "train_split.npz"),
            batch_array,
            labels
        )
        np.save(
            os.path.join(dataset_dir, "train_color.npy"),
            batch_array
        )

        gray_array = np.zeros(batch_array.shape)
        R = batch_array[:, 0, :, :] 
        G = batch_array[:, 1, :, :] 
        B = batch_array[:, 2, :, :] 

        avg_channel = R * 0.299 + G * 0.587 + B * 0.114
        for i in range(3):
            gray_array[:, i, :, :] = avg_channel
        np.save(
            os.path.join(dataset_dir, "train_gray.npy"),
            gray_array
        )

        dataset = datasets.CIFAR10(
            root=dataset_dir,
            train=False,
            transform=transforms.ToTensor()
        )
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        batch_array, labels = next(iter(loader))

        batch_array = batch_array.numpy()
        labels = labels.numpy()

        np.savez(
            os.path.join(dataset_dir, "test_split.npz"),
            batch_array,
            labels
        )
        np.save(
            os.path.join(dataset_dir, "test_color.npy"),
            batch_array
        )

        gray_array = np.zeros(batch_array.shape)
        R = batch_array[:, 0, :, :] 
        G = batch_array[:, 1, :, :] 
        B = batch_array[:, 2, :, :] 

        avg_channel = R * 0.299 + G * 0.587 + B * 0.114
        for i in range(3):
            gray_array[:, i, :, :] = avg_channel
        np.save(
            os.path.join(dataset_dir, "test_gray.npy"),
            gray_array
        )

def process_samples(args):
    save_file = np.load(args.sample_file, allow_pickle=True)

    samples = save_file['arr_0'] # shape = num_samples x height x width x n_channel
    samples = np.transpose(samples, (0, 3, 1, 2))
    samples = samples.astype(np.float64) / 255.0

    np.save(
        args.sample_file.replace(".npz", "_images.npy"),
        samples
    )


def get_args():
    parser = argparse.ArgumentParser("Measure generation samplng distribution compared to training distribution")

    parser.add_argument('--dataset', type=str, help="dataset used for experiment")
    parser.add_argument('--train-color', type=float, help="portion of training distribution that are colored images")
    parser.add_argument('--train-gray', type=float, help="portion of training distribution that are gradyscaled images")

    parser.add_argument('--sample-color', action="store_true", help="only generate color samples")
    parser.add_argument('--sample-gray', action="store_true", help="only generate gray samples")
    parser.add_argument('--num-samples', type=int, help="number of samples")
    
    parser.add_argument('--date', type=str, help="experiment date for logging purpose")
    parser.add_argument('--diffusion-config', type=str, default="UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512", help="diffusion model configuration, currently set to be default parameters")

    parser.add_argument('--proc-real', action="store_true", help="whether to only process real distribution")

    args = parser.parse_args()

    args.log_dir = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs", 
                        args.date, args.dataset, "color{}_gray{}".format(args.train_color, args.train_gray), 
                        args.diffusion_config)

    if args.sample_color:
        sample_str = "color"
    elif args.sample_gray:
        sample_str = "gray"
    else:
        raise NotImplementedError
    args.sample_file = os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/logs", 
                        args.date, args.dataset, "color{}_gray{}".format(args.train_color, args.train_gray), 
                        args.diffusion_config, "samples", 
                        "{}_color{}_gray{}_epoch_{}_num{}_{}.npz".format(args.dataset, args.train_color, args.train_gray, 950, args.num_samples, sample_str))
    
    return args


def main():
    args = get_args()

    if args.proc_real:
        process_real_dataset(args.dataset)
        return

    print("Evaluating training distribution with {} colored and {} grayscaled".format(args.train_color, args.train_gray))
    process_samples(args)


if __name__ == "__main__":
    main()