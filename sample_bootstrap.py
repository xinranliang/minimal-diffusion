import os
import cv2
import copy
import math
import argparse
import numpy as np
from tqdm import tqdm
from easydict import EasyDict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset.data import get_metadata, get_dataset
from model.diffusion import GuassianDiffusion
import model.unets as unets
import utils


# sample code
def sample_N_images(
    N,
    model,
    diffusion,
    xT=None,
    sampling_steps=250,
    batch_size=64,
    num_channels=3,
    image_size=32,
    num_classes=None,
    args=None,
):
    """use this function to sample any number of images from a given
        diffusion model and diffusion process.

    Args:
        N : Number of images
        model : Diffusion model
        diffusion : Diffusion process
        xT : Starting instantiation of noise vector.
        sampling_steps : Number of sampling steps.
        batch_size : Batch-size for sampling.
        num_channels : Number of channels in the image.
        image_size : Image size (assuming square images).
        num_classes : Number of classes in the dataset (needed for class-conditioned models)
        args : All args from the argparser.

    Returns: Numpy array with N images and corresponding labels.
    """
    samples, labels, num_samples = [], [], 0
    num_processes, group = dist.get_world_size(), dist.group.WORLD

    while num_samples < N:
        if xT is None:
            xT = (
                torch.randn(batch_size, num_channels, image_size, image_size)
                .float()
                .to(args.device)
            )
        if args.class_cond:
            y = torch.randint(num_classes, (len(xT),), dtype=torch.int64).to(
                args.device
            )
        else:
            y = None
        gen_images = diffusion.sample_from_reverse_process(
            model, xT, sampling_steps, {"y": y}, args.ddim
        )
        samples_list = [torch.zeros_like(gen_images) for _ in range(num_processes)]
        if args.class_cond:
            labels_list = [torch.zeros_like(y) for _ in range(num_processes)]
            dist.all_gather(labels_list, y, group)
            labels.append(torch.cat(labels_list).detach().cpu().numpy())

        dist.all_gather(samples_list, gen_images, group)
        samples.append(torch.cat(samples_list).detach().cpu().numpy())
        num_samples += len(xT) * num_processes

    samples = np.concatenate(samples).transpose(0, 2, 3, 1)[:N]
    samples = (127.5 * (samples + 1)).astype(np.uint8)
    return (samples, np.concatenate(labels) if args.class_cond else None)


def get_args():
    parser = argparse.ArgumentParser("Minimal implementation of diffusion models")
    # diffusion model
    parser.add_argument("--arch", type=str, help="Neural network architecture")
    parser.add_argument(
        "--class-cond",
        action="store_true",
        default=False,
        help="train class-conditioned diffusion model",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=1000,
        help="Number of timesteps in diffusion process",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=250,
        help="Number of timesteps in diffusion process",
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        default=False,
        help="Sampling using DDIM update step",
    )

    parser.add_argument("--dataset", type=str)

    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch-size per gpu"
    )
    parser.add_argument("--pretrained-ckpt", type=str, help="Pretrained model ckpt directory")
    parser.add_argument("--ckpt-name", type=str, help="name of cpretrained ckpt for sampling in logging")
    parser.add_argument("--delete-keys", nargs="+", help="Pretrained model ckpt")
    parser.add_argument(
        "--sampling-only",
        action="store_true",
        default=True,
        help="No training, just sample images (will save them in --save-dir)",
    )
    parser.add_argument(
        "--num-sampled-images",
        type=int,
        default=50000,
        help="Number of images required to sample from the model",
    )
    parser.add_argument("--num-bootstrap", type=int, help="number of times in order to get confidence interval from empirical distirbution")
    parser.add_argument("--threshold", type=float, default=1.0, help="threshold of seperating color and gray in terms of channel-wise std")

    # misc
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--seed", default=112233, type=int)

    # setup
    args = parser.parse_args()

    return args


def main(args):
    metadata = get_metadata(args.dataset, None, None)

    # distribute data parallel
    torch.backends.cudnn.benchmark = True
    args.device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    if args.local_rank == 0:
        print(args)
    
    # Creat model and diffusion process
    model = unets.__dict__[args.arch](
        image_size=metadata.image_size,
        in_channels=metadata.num_channels,
        out_channels=metadata.num_channels,
        num_classes=metadata.num_classes if args.class_cond else None,
    ).to(args.device)
    if args.local_rank == 0:
        print(
            "We are assuming that model input/ouput pixel range is [-1, 1]. Please adhere to it."
        )
    diffusion = GuassianDiffusion(args.diffusion_steps, args.device)

    # load pre-trained model
    print(f"Loading pretrained model from {args.pretrained_ckpt}")
    d = utils.fix_legacy_dict(torch.load(args.pretrained_ckpt, map_location=args.device))
    dm = model.state_dict()
    if args.delete_keys:
        for k in args.delete_keys:
            print(
                f"Deleting key {k} becuase its shape in ckpt ({d[k].shape}) doesn't match "
                + f"with shape in model ({dm[k].shape})"
            )
            del d[k]
    model.load_state_dict(d, strict=False)
    print(
        f"Mismatched keys in ckpt and model: ",
        set(d.keys()) ^ set(dm.keys()),
    )
    print(f"Loaded pretrained model from {args.pretrained_ckpt}")

    # distributed training
    ngpus = torch.cuda.device_count()
    if ngpus > 1:
        if args.local_rank == 0:
            print(f"Using distributed training on {ngpus} gpus.")
        args.batch_size = args.batch_size // ngpus
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    # sampling
    if args.sampling_only:
        ratio_colors = []
        ratio_grays = []

        with tqdm(total=args.num_bootstrap) as pbar:
            for bootstrap in range(args.num_bootstrap):
                sampled_images, labels = sample_N_images(
                    args.num_sampled_images,
                    model,
                    diffusion,
                    None,
                    args.sampling_steps,
                    args.batch_size,
                    metadata.num_channels,
                    metadata.image_size,
                    metadata.num_classes,
                    args,
                )
                # shape: num_samples x height x width x num_channel
                channel_std = np.std(sampled_images.astype(np.float64), axis=-1) # num_samples x height x width
                channel_std = channel_std.reshape((channel_std.shape[0], channel_std.shape[1] * channel_std.shape[2])) # num_samples x (height x width)
                channel_std = np.mean(channel_std, axis=-1) # num_samples x 1

                num_color = np.sum(channel_std >= args.threshold)
                num_gray = np.sum(channel_std < args.threshold)
                ratio_color, ratio_gray = num_color / args.num_sampled_images * 100, num_gray / args.num_sampled_images * 100

                ratio_colors.append(ratio_color)
                ratio_grays.append(ratio_gray)

                if args.local_rank == 0:
                    pbar.update(1)
    
    # compute confidence interval from empirical distribution
    ratio_colors = np.array(ratio_colors, dtype=np.float64)
    ratio_grays = np.array(ratio_grays, dtype=np.float64)

    if args.local_rank == 0:
        print("Aggregating bootstrap statistics for ckpt {} from {} samples:".format(args.ckpt_name, args.num_bootstrap))

        print("Aggregating summary statistics: mean (std)")

        # mean, standard deviation
        print("Color percentage: {} ({})".format(np.mean(ratio_colors), np.std(ratio_colors)))
        print("Gray percentage: {} ({})".format(np.mean(ratio_grays), np.std(ratio_grays)))


if __name__ == "__main__":
    args = get_args()
    main(args)