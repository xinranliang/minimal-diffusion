import numpy as np
import pickle
import cv2
from PIL import Image
import os
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from cleanfid import fid

from precision_recall import compute_precision_recall


def process_real(dataset, resolution, num_channels=3, date="none", mode="none"):
    r"""
    Function to save real images train split into a specific folder, used for computing FID statistics

    Input:
    - dataset: name of dataset, by default using train split
    - resolution: resolution of images
    - num_channels: default RGB channels

    Output:
    - save_folder: path directory to saved images
    """

    root_dir = os.path.join("./datasets", dataset)
    image_index = None
    if args.date != "none":
        if dataset == "cifar10":
            if mode == "color":
                with open(os.path.join(root_dir, "color_gray_split", date, "color15000_gray0_index.pkl"), "rb") as f:
                    image_index = pickle.load(f)["color_index"]
            elif mode == "gray":
                with open(os.path.join(root_dir, "color_gray_split", date, "color0_gray15000_index.pkl"), "rb") as f:
                    image_index = pickle.load(f)["gray_index"]
        
        elif dataset == "mix-cifar10-imagenet":
            root_dir = "./datasets/cifar10-imagenet" # consistent with logging
            if mode == "color":
                with open(os.path.join(root_dir, "index_split", date, "color15000_gray0_index.pkl"), "rb") as f:
                    image_index = pickle.load(f)["color_index"]
            elif mode == "gray":
                with open(os.path.join(root_dir, "index_split", date, "gray15000_color0_index.pkl"), "rb") as f:
                    image_index = pickle.load(f)["gray_index"]
    
    if dataset == "cifar10":
        root_dir = os.path.join(root_dir, "cifar-10-batches-py")

        # train split
        if args.date == "none":
            if mode == "color":
                save_folder = os.path.join("./logs/cifar10_color")
            elif mode == "gray":
                save_folder = os.path.join("./logs/cifar10_gray")
        else:
            if mode == "color":
                save_folder = os.path.join("./logs", args.date, "cifar10_color")
            elif mode == "gray":
                save_folder = os.path.join("./logs", args.date, "cifar10_gray")
        os.makedirs(save_folder, exist_ok=True)

        img_idx = 0
        for num_batch in range(1, 6):
            batch_file = os.path.join(root_dir, "data_batch_{}".format(int(num_batch)))
            with open(batch_file, "rb") as f:
                batch_dict = pickle.load(f, encoding="bytes")
            
            # data, label
            images = batch_dict[b"data"]
            labels = batch_dict[b"labels"]
            images = np.array(images, dtype=np.uint8)
            images = images.reshape([-1, num_channels, resolution, resolution])
            images = np.transpose(images, (0, 2, 3, 1))

            for idx in range(images.shape[0]):
                if mode == "color":
                    image = Image.fromarray(images[idx], "RGB")
                elif mode == "gray":
                    image = Image.fromarray(rgb_to_grayscale(images[idx]), "RGB")
                
                if image_index is not None:
                    if img_idx in image_index:
                        image.save(os.path.join(save_folder, "train_%05d.png" % (img_idx)))
                else:
                    image.save(os.path.join(save_folder, "train_%05d.png" % (img_idx)))
                img_idx += 1
        
        return save_folder
    
    elif dataset == "mix-cifar10-imagenet":
        if args.date == "none":
            if mode == "color":
                save_folder = os.path.join("./logs/mix-cifar10-imagenet_color")
            elif mode == "gray":
                save_folder = os.path.join("./logs/mix_cifar10-imagenet_gray")
        else:
            if mode == "color":
                save_folder = os.path.join("./logs", args.date, "mix-cifar10-imagenet_color")
            elif mode == "gray":
                save_folder = os.path.join("./logs", args.date, "mix-cifar10-imagenet_gray")
        os.makedirs(save_folder, exist_ok=True)

        full_data = datasets.ImageFolder(
            root = "/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/train",
            transform = None,
            target_transform = None
        )

        if image_index is not None:
            for img_idx in image_index:
                image, label = full_data.__getitem__(img_idx)
                if mode == "gray":
                    image = image.convert("L")
                image.save(os.path.join(save_folder, "%s.png" % (img_idx)))



def array_to_image(path, num_images, folder="./logs/temp"):
    if path.endswith(".npy"):
        images = np.load(path)
    elif path.endswith("npz"):
        images = np.load(path, allow_pickle=True)["arr_0"]
    else:
        raise ValueError(f"Unrecognized file type: {path}")
    
    if images.min() >= 0 and images.max() <= 1:
        images = (images * 255).astype("uint8")
    elif images.min() >= -1 and images.max() <= 1:
        images = (127.5 * (images + 1)).astype("uint8")
    else:
        assert images.min() >= 0 and images.max() <= 255
    
    assert len(images.shape) == 4, "Images must be a batch"
    assert num_images <= images.shape[0], "Not enough available samples to evaluate!"
    for idx in range(num_images):
        # image = Image.fromarray(images[idx], "RGB")
        # image.save(os.path.join(folder, "sample_%05d.png" % (idx)))
        cv2.imwrite(os.path.join(folder, "sample_%05d.png" % (idx)), images[idx, :, :, ::-1])


def main(args):
    real_path = None
    if args.save_real:
        if args.date == "none": # full 50k metrics
            real_path = process_real(args.dataset, args.resolution, mode=args.real_mode)
        else:
            real_path = process_real(args.dataset, args.resolution, date=args.date, mode=args.real_mode)
        return
    else: 
        if args.real_mode == "color":
            if args.date == "none": # full 50k metrics
                real_path = f"./logs/{args.dataset}_color"
            else:
                real_path = os.path.join("./logs", args.date, f"{args.dataset}_color")
                os.makedirs(real_path, exist_ok=True)
        elif args.real_mode == "gray":
            if args.date == "none": # full 50k metrics
                real_path = f"./logs/{args.dataset}_gray"
            else:
                real_path = os.path.join("./logs", args.date, f"{args.dataset}_gray")
                os.makedirs(real_path, exist_ok=True)
    
    # construct image folder
    if args.date == "none": # full 50k metrics
        array_to_image(args.fake, args.num_samples)
    else:
        fake_path = os.path.join("./logs", args.date, "tmp")
        os.makedirs(fake_path, exist_ok=True)
        array_to_image(args.fake, args.num_samples, folder=fake_path)

    # compute fid
    if args.real_mode == "color":
        if args.date == "none": # full 50k metrics
            fid_score = fid.compute_fid("./logs/temp/", mode=args.mode, dataset_name=args.dataset, dataset_res=args.resolution, dataset_split="train", batch_size = args.batch_size, num_workers = args.num_gpus * 4)
        else:
            fid_score = fid.compute_fid(real_path, fake_path, mode=args.mode, batch_size = args.batch_size, num_workers = args.num_gpus * 4)
        print(f"{args.mode}-fid score with folder-wise statistics: {fid_score:.3f}")

    if args.real_mode == "gray":
        if args.date == "none": # full 50k metrics
            fid_score = fid.compute_fid(real_path, "./logs/temp/", mode=args.mode, batch_size = args.batch_size, num_workers = args.num_gpus * 4)
        else:
            fid_score = fid.compute_fid(real_path, fake_path, mode=args.mode, batch_size = args.batch_size, num_workers = args.num_gpus * 4)
        print(f"{args.mode}-fid score with folder-wise statistics: {fid_score:.3f}")
    
    # compute precision recall
    # check: fdir1 is real folder path and fdir2 is fake folder path
    if args.date == "none": # full 50k metrics
        precision_recall = compute_precision_recall(real_path, "./logs/temp/", mode=args.mode, batch_size = args.batch_size, num_workers = args.num_gpus * 4)
    else:
        precision_recall = compute_precision_recall(real_path, fake_path, mode=args.mode, batch_size = args.batch_size, num_workers = args.num_gpus * 4)
    precision = precision_recall["precision"]
    recall = precision_recall["recall"]
    print(f"{args.mode}-precision: {precision:.3f}")
    print(f"{args.mode}-recall: {recall:.3f}")

    return


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, required=True, help='real dataset to evaluate')
    parser.add_argument('--save-real', action="store_true", help="whether to save real dataset into folders")
    parser.add_argument('--real-mode', type=str, choices=["color", "gray"])
    parser.add_argument('--date', type=str, default="none", help="specify date of experiments, if there is any, to handle varied runs of sampling subsets")
    parser.add_argument('--num-samples', type=int, default=50000, help="number of samples used to compute metrics, handle different sizes of domain images")
    parser.add_argument('--fake', type=str, required=True, help=('Path to the generated images'))
    parser.add_argument('--resolution', type=int, required=True, help='image resolution to compute metrics')
    parser.add_argument('--mode', type=str, default="clean", choices=["clean", "legacy_pytorch", "legacy_tensorflow"])
    parser.add_argument('--batch-size', type=int, default=50, help="batch size to extract features")
    parser.add_argument("--num-gpus", type=int, help='number of GPUs for evaluation')
    args = parser.parse_args()

    if args.num_gpus > 0 and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    print(args)
    
    main(args)