import os
import cv2
from PIL import Image
import copy
import math
import argparse
import numpy as np
from tqdm import tqdm
import torch

import utils
import model.unets as unets
from model.diffusion import GuassianDiffusion

def sample_N_images(
    N,
    model,
    diffusion,
    xT,
    sampling_steps=250,
    batch_size=64,
    num_channels=3,
    image_size=32,
    num_classes=None,
    args=None,
):
    samples, labels, num_samples = [], [], 0
    # fix latent
    assert xT is not None
    with tqdm(total=math.ceil(N / args.batch_size)) as pbar:
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
                model, xT, sampling_steps, {"y": y}, args.ddim, args.classifier_free_w
            )

            if args.sampling_mode == "color":
                # select samples
                channel_std = torch.std(gen_images, dim=1).reshape(gen_images.shape[0], gen_images.shape[2] * gen_images.shape[3]).mean(-1) # (num_samples x 1 x height x width)
                index_select = torch.nonzero(channel_std >= args.threshold["norm_val"]).reshape(-1)
                select_samples = torch.index_select(gen_images, dim=0, index=index_select)
                # select labels
                select_labels = torch.index_select(y, dim=0, index=index_select)
            elif args.sampling_mode == "gray":
                # select samples
                channel_std = torch.std(gen_images, dim=1).reshape(gen_images.shape[0], gen_images.shape[2] * gen_images.shape[3]).mean(-1) # (num_samples x 1 x height x width)
                index_select = torch.nonzero(channel_std < args.threshold["norm_val"]).reshape(-1)
                select_samples = torch.index_select(gen_images, dim=0, index=index_select)
                # select labels
                select_labels = torch.index_select(y, dim=0, index=index_select)
            elif args.sampling_mode == "none":
                select_samples = gen_images
                select_labels = y
            else:
                # not implemented right now
                raise NotImplementedError

            if args.class_cond:
                labels.append(select_labels.detach().cpu().numpy())

            samples.append(select_samples.detach().cpu().numpy())
            num_samples += index_select.shape[0]
            pbar.update(1)
    samples = np.concatenate(samples).transpose(0, 2, 3, 1)[:N] # shape = num_samples x height x width x n_channel
    samples = (127.5 * (samples + 1)).astype(np.uint8)

    data_dict = {
        "arr_0": samples,
        "arr_1": np.concatenate(labels)[:N] if args.class_cond else None
    }
    return data_dict


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
    parser.add_argument(
        "--classifier-free-w",
        type=float,
        default=0.0,
        help="Weight of classfiier-free guidance in sampling process",
    )

    # dataset
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data-dir", type=str, default="./dataset/")

    # optimizer
    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch-size per gpu"
    )

    # sampling/finetuning
    parser.add_argument(
        "--sampling-only",
        action="store_true",
        default=False,
        help="No training, just sample images (will save them in --save-dir)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of images required to sample from the model",
    )
    parser.add_argument(
        "--sampling-mode",
        type=str,
        choices=["none", "color", "gray", "cifar", "imagenet"],
        help="specify domain for synthetic samples"
    )

    # visualize
    parser.add_argument('--num-visualize', type=int, default=100, help="number of images to visualize, should be square number n^2")

    # misc
    parser.add_argument("--date", type=str)
    parser.add_argument("--save-dir", type=str, default="./trained_models/")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--seed", default=112233, type=int)

    # setup
    args = parser.parse_args()

    return args


def sample_fixed_latent(args, model, diffusion):
    """
    generate fixed latent
    """

    # list of model ckeckpoint path

    if args.dataset == "cifar10":
        model_ckpts = [
            os.path.join(args.save_dir, "color15000_gray0", args.diffusion_config, "ckpt", "epoch_3333_ema_0.9995.pth"),
            os.path.join(args.save_dir, "color15000_gray5000", args.diffusion_config, "ckpt", "epoch_2499_ema_0.9995.pth"),
            os.path.join(args.save_dir, "color15000_gray10000", args.diffusion_config, "ckpt", "epoch_1999_ema_0.9995.pth"),
            os.path.join(args.save_dir, "color15000_gray15000", args.diffusion_config, "ckpt", "epoch_1666_ema_0.9995.pth"),
            os.path.join(args.save_dir, "color15000_gray20000", args.diffusion_config, "ckpt", "epoch_1499_ema_0.9995.pth"),
            os.path.join(args.save_dir, "color15000_gray25000", args.diffusion_config, "ckpt", "epoch_1249_ema_0.9995.pth"),
            os.path.join(args.save_dir, "color15000_gray30000", args.diffusion_config, "ckpt", "epoch_1110_ema_0.9995.pth"),
            os.path.join(args.save_dir, "color0.3_gray0.7", args.diffusion_config, "ckpt", "epoch_1000_ema_0.9995.pth"),
        ]
        file_paths = [
            os.path.join(args.save_dir, "color15000_gray0", args.diffusion_config, "visualize", f"color15k_gray0k_num{args.num_visualize}.png"),
            os.path.join(args.save_dir, "color15000_gray5000", args.diffusion_config, "visualize", f"color15k_gray5k_num{args.num_visualize}.png"),
            os.path.join(args.save_dir, "color15000_gray10000", args.diffusion_config, "visualize", f"color15k_gray10k_num{args.num_visualize}.png"),
            os.path.join(args.save_dir, "color15000_gray15000", args.diffusion_config, "visualize", f"color15k_gray15k_num{args.num_visualize}.png"),
            os.path.join(args.save_dir, "color15000_gray20000", args.diffusion_config, "visualize", f"color15k_gray20k_num{args.num_visualize}.png"),
            os.path.join(args.save_dir, "color15000_gray15000", args.diffusion_config, "visualize", f"color15k_gray25k_num{args.num_visualize}.png"),
            os.path.join(args.save_dir, "color15000_gray30000", args.diffusion_config, "visualize", f"color15k_gray30k_num{args.num_visualize}.png"),
            os.path.join(args.save_dir, "color0.3_gray0.7", args.diffusion_config, "visualize", f"color15k_gray35k_num{args.num_visualize}.png"),
        ]

    elif args.dataset == "mix-cifar10-imagenet":
        model_ckpts = [
            os.path.join(args.save_dir, "color15000_gray0", args.diffusion_config, "ckpt", "epoch_4000_ema_0.9995.pth"),
            os.path.join(args.save_dir, "color15000_gray30000", args.diffusion_config, "ckpt", "epoch_1419_ema_0.9995.pth"),
            os.path.join(args.save_dir, "color15000_gray60000", args.diffusion_config, "ckpt", "epoch_849_ema_0.9995.pth"),
            os.path.join(args.save_dir, "color15000_gray90000", args.diffusion_config, "ckpt", "epoch_609_ema_0.9995.pth"),
            os.path.join(args.save_dir, "color15000_gray120000", args.diffusion_config, "ckpt", "epoch_474_ema_0.9995.pth"),
            os.path.join(args.save_dir, "color15000_gray150000", args.diffusion_config, "ckpt", "epoch_389_ema_0.9995.pth"),
            os.path.join(args.save_dir, "color15000_gray180000", args.diffusion_config, "ckpt", "epoch_329_ema_0.9995.pth"),
            os.path.join(args.save_dir, "color15000_gray210000", args.diffusion_config, "ckpt", "epoch_284_ema_0.9995.pth"),
            os.path.join(args.save_dir, "color15000_gray240000", args.diffusion_config, "ckpt", "epoch_249_ema_0.9995.pth"),
        ]
        file_paths = [
            os.path.join(args.save_dir, "color15000_gray0", args.diffusion_config, "visualize", f"color15k_gray0k_num{args.num_visualize}.png"),
            os.path.join(args.save_dir, "color15000_gray30000", args.diffusion_config, "visualize", f"color15k_gray30k_num{args.num_visualize}.png"),
            os.path.join(args.save_dir, "color15000_gray60000", args.diffusion_config, "visualize", f"color15k_gray60k_num{args.num_visualize}.png"),
            os.path.join(args.save_dir, "color15000_gray90000", args.diffusion_config, "visualize", f"color15k_gray90k_num{args.num_visualize}.png"),
            os.path.join(args.save_dir, "color15000_gray120000", args.diffusion_config, "visualize", f"color15k_gray120k_num{args.num_visualize}.png"),
            os.path.join(args.save_dir, "color15000_gray150000", args.diffusion_config, "visualize", f"color15k_gray150k_num{args.num_visualize}.png"),
            os.path.join(args.save_dir, "color15000_gray180000", args.diffusion_config, "visualize", f"color15k_gray180k_num{args.num_visualize}.png"),
            os.path.join(args.save_dir, "color15000_gray210000", args.diffusion_config, "visualize", f"color15k_gray210k_num{args.num_visualize}.png"),
            os.path.join(args.save_dir, "color15000_gray240000", args.diffusion_config, "visualize", f"color15k_gray240k_num{args.num_visualize}.png"),
        ]

    assert len(model_ckpts) == len(file_paths)

    # generate a fixed latent
    xT = (torch.randn(args.batch_size, 3, 32, 32).float().to(args.device))

    for index in range(len(model_ckpts)):
        # load model checkpoint
        pretrained_checkpoint = model_ckpts[index]
        d = utils.fix_legacy_dict(torch.load(pretrained_checkpoint, map_location=args.device))
        model.load_state_dict(d)
        print(f"Loaded pretrained model from {pretrained_checkpoint}")

        # deepcopy fixed latent and sample from there
        xT_temp = xT.clone().detach()
        data_dict = sample_N_images(
             args.num_samples,
             model,
             diffusion,
             xT_temp,
             args.sampling_steps,
             args.batch_size,
             3,
             32,
             args.num_classes,
             args
        )

        # save figure
        os.makedirs(os.path.join(*file_paths[index].split("/")[:-1]), exist_ok=True)
        utils.arr_to_img(data_dict, file_paths[index], args, random_sample=False)
    
    return


def main(args):
    
    # logger
    args.save_dir = os.path.join(args.save_dir, args.date, args.dataset)
    if args.class_cond:
        args.num_classes = 10

    args.device = "cuda:{}".format(args.local_rank)
    if args.local_rank == 0:
        print(args)
    
    # Creat model and diffusion process
    model = unets.__dict__[args.arch](
        image_size=32,
        in_channels=3,
        out_channels=3,
        num_classes=args.num_classes if args.class_cond else None,
    ).to(args.device)
    if args.local_rank == 0:
        print(
            "We are assuming that model input/ouput pixel range is [-1, 1]. Please adhere to it."
        )
    diffusion = GuassianDiffusion(args.diffusion_steps, args.device)

    args.diffusion_config = "UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1"

    # threshold
    args.threshold = {}
    if args.dataset == "cifar10":
        args.threshold["norm_val"] = 0.006
        args.threshold["pixel_val"] = 0.74
    elif args.dataset == "mix-cifar10-imagenet":
        args.threshold["norm_val"] = 0.003
        args.threshold["pixel_val"] = 0.86
    
    sample_fixed_latent(args, model, diffusion)


if __name__ == "__main__":
    args = get_args()
    main(args)
