import os
import cv2
import copy
import math
import argparse
import numpy as np
from tqdm import tqdm
from easydict import EasyDict

import torch
from torchvision import datasets, transforms
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset.data import get_metadata, get_dataset
from dataset.utils import ArrayToImageLabel
from model.diffusion import GuassianDiffusion, train_one_epoch, train_one_epoch_gender, sample_N_images, sample_N_images_nodist, sample_color_images, sample_gray_images, sample_N_images_cond, sample_N_images_classifier
import model.unets as unets
from domain_classifier.cifar_imagenet import DomainClassifier as DC_cifar_imgnet
from domain_classifier.mnist_flip import DomainClassifier as DC_mnist, count_flip
import utils
from metrics.color_gray import count_colorgray, compute_colorgray


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
        "--class-cond-dropout",
        type=float,
        default=0.0,
        help="Probability of droping class labels in training"
    )

    # dataset
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data-dir", type=str, default="./dataset/")

    parser.add_argument("--fix", type=str, choices=["total", "color", "gray", "none", "half"], default="none", help="specify how to split training distribution")

    # color-gray domain
    parser.add_argument("--color", type=int, required=False, help="ratio or number of training distribution to be turned into colored images")
    parser.add_argument("--grayscale", type=int, required=False, help="ratio or number of training distribution to be turned into grayscale images")

    # cifar10-imagenet domain
    parser.add_argument("--num-cifar", type=float, required=False, help="number of cifar10 images used for training")
    parser.add_argument("--num-imagenet", type=float, required=False, help="number of imagenet images used for training")
    parser.add_argument("--num-baseline", type=int, required=False, help="number of baseline images used for training, mixed from cifar10 and imagenet")

    # mnist flip left-right domain
    parser.add_argument("--flip-left", type=float, required=False, help="ratio of mnist images flipped to left")
    parser.add_argument("--flip-right", type=float, required=False, help="ratio of mnist images flipped to right")

    # cifar superclass domain
    parser.add_argument("--semantic-group", type=str, required=False, choices=["similar", "distinct"], help="how to group classes from semantic domains")
    parser.add_argument("--front-ratio", type=float, required=False, help="ratio of classes belong to first 5 indices")
    parser.add_argument("--back-ratio", type=float, required=False, help="ratio of classes belong to last 5 indices")

    # celeba/fairface gender domain
    parser.add_argument("--female-ratio", type=float, required=False, help="ratio of female images per conditioning class")
    parser.add_argument("--male-ratio", type=float, required=False, help="ratio of male images per conditioning class")

    # optimizer
    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch-size per gpu"
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--ema_w", type=float, default=0.9995)
    parser.add_argument("--ckpt-sample-freq", type=int, default=50)

    # sampling/finetuning
    parser.add_argument("--pretrained-ckpt", type=str, help="Pretrained model ckpt directory")
    parser.add_argument("--ckpt-name", type=str, help="name of cpretrained ckpt for sampling in logging")
    parser.add_argument("--delete-keys", nargs="+", help="Pretrained model ckpt")
    parser.add_argument("--domain-classifier", type=str, required=False, help="pretrained classifier to detect synthetic samples with certain attributes")

    parser.add_argument(
        "--sampling-only",
        action="store_true",
        default=False,
        help="No training, just sample images (will save them in --save-dir)",
    )
    parser.add_argument(
        "--num-sampled-images",
        type=int,
        default=50000,
        help="Number of images required to sample from the model",
    )
    
    # misc
    parser.add_argument("--date", type=str)
    parser.add_argument("--save-dir", type=str, default="./trained_models/")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--seed", default=112233, type=int)

    # setup
    args = parser.parse_args()

    return args


def main(args):
    
    # logger
    args.save_dir = os.path.join(args.save_dir, args.date, args.dataset)

    metadata = get_metadata(
        name=args.dataset, date=args.date,
        fix=args.fix, color=args.color, grayscale=args.grayscale,
        num_cifar=args.num_cifar, num_imagenet=args.num_imagenet, num_train_baseline=args.num_baseline,
        flip_left=args.flip_left, flip_right=args.flip_right,
        semantic_group=args.semantic_group, front_ratio=args.front_ratio, back_ratio=args.back_ratio,
        female_ratio=args.female_ratio, male_ratio=args.male_ratio,
    )

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
        prob_drop_cond=args.class_cond_dropout
    ).to(args.device)
    if args.local_rank == 0:
        print(
            "We are assuming that model input/ouput pixel range is [-1, 1]. Please adhere to it."
        )
    diffusion = GuassianDiffusion(args.diffusion_steps, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # load pre-trained model
    if args.pretrained_ckpt:
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
    
    # logging
    if "cifar10" in args.dataset:
        if args.fix == "total" or args.fix == "color":
            log_dir = os.path.join(
                    args.save_dir, 
                    "color{}_gray{}".format(args.color, args.grayscale),
                    "{}_diffusionstep_{}_samplestep_{}_condition_{}_lr_{}_bs_{}_dropprob_{}".format(
                        args.arch, args.diffusion_steps, args.sampling_steps, args.class_cond, args.lr, args.batch_size * ngpus, args.class_cond_dropout
                    )
                    )
        elif args.fix == "gray":
            log_dir = os.path.join(
                    args.save_dir, 
                    "gray{}_color{}".format(args.grayscale, args.color),
                    "{}_diffusionstep_{}_samplestep_{}_condition_{}_lr_{}_bs_{}_dropprob_{}".format(
                        args.arch, args.diffusion_steps, args.sampling_steps, args.class_cond, args.lr, args.batch_size * ngpus, args.class_cond_dropout
                    )
                    )
        elif args.fix == "none" and args.dataset == "cifar10-imagenet":
            log_dir = os.path.join(
                    args.save_dir, 
                    "cifar{}_imagenet{}".format(args.num_cifar10, args.num_imagenet),
                    "{}_diffusionstep_{}_samplestep_{}_condition_{}_lr_{}_bs_{}_dropprob_{}".format(
                        args.arch, args.diffusion_steps, args.sampling_steps, args.class_cond, args.lr, args.batch_size * ngpus, args.class_cond_dropout
                    )
                    )
        elif args.fix == "none" and args.dataset == "mix-cifar10-imagenet":
            log_dir = os.path.join(
                    args.save_dir, 
                    "baseline_num{}".format(args.num_baseline),
                    "{}_diffusionstep_{}_samplestep_{}_condition_{}_lr_{}_bs_{}_dropprob_{}".format(
                        args.arch, args.diffusion_steps, args.sampling_steps, args.class_cond, args.lr, args.batch_size * ngpus, args.class_cond_dropout
                    )
                    )
        elif args.fix == "half":
            if args.color == args.grayscale:
                log_dir = os.path.join(
                        args.save_dir,
                        "half{}".format(args.color),
                        "{}_diffusionstep_{}_samplestep_{}_condition_{}_lr_{}_bs_{}_dropprob_{}".format(
                            args.arch, args.diffusion_steps, args.sampling_steps, args.class_cond, args.lr, args.batch_size * ngpus, args.class_cond_dropout
                        )
                )
            else:
                assert args.color == 0 or args.grayscale == 0
                if args.color == 0:
                    log_dir = os.path.join(
                            args.save_dir,
                            "half_gray{}".format(args.grayscale),
                            "{}_diffusionstep_{}_samplestep_{}_condition_{}_lr_{}_bs_{}_dropprob_{}".format(
                                args.arch, args.diffusion_steps, args.sampling_steps, args.class_cond, args.lr, args.batch_size * ngpus, args.class_cond_dropout
                            )
                    )
                if args.grayscale == 0:
                    log_dir = os.path.join(
                            args.save_dir,
                            "half_color{}".format(args.color),
                            "{}_diffusionstep_{}_samplestep_{}_condition_{}_lr_{}_bs_{}_dropprob_{}".format(
                                args.arch, args.diffusion_steps, args.sampling_steps, args.class_cond, args.lr, args.batch_size * ngpus, args.class_cond_dropout
                            )
                    )
        else:
            raise NotImplementedError
    
    elif "mnist" in args.dataset:
        log_dir = os.path.join(
                args.save_dir, 
                "left{}_right{}".format(args.flip_left, args.flip_right),
                "{}_diffusionstep_{}_samplestep_{}_condition_{}_lr_{}_bs_{}_dropprob_{}".format(
                    args.arch, args.diffusion_steps, args.sampling_steps, args.class_cond, args.lr, args.batch_size * ngpus, args.class_cond_dropout
                        )
                )
    
    elif "cifar-super" in args.dataset:
        log_dir = os.path.join(
            args.save_dir,
            "group_{}".format(args.semantic_group),
            "front{}_back{}".format(args.front_ratio, args.back_ratio),
            "{}_diffusionstep_{}_samplestep_{}_condition_{}_lr_{}_bs_{}_dropprob_{}".format(
                args.arch, args.diffusion_steps, args.sampling_steps, args.class_cond, args.lr, args.batch_size * ngpus, args.class_cond_dropout
                )
        ) 
    
    elif "cifar-imagenet" in args.dataset:
        log_dir = os.path.join(
            args.save_dir, 
            "cifar{}_imagenet{}_numclass{}".format(args.num_cifar, args.num_imagenet, metadata.num_classes),
            "{}_diffusionstep_{}_samplestep_{}_condition_{}_lr_{}_bs_{}_dropprob_{}".format(
                args.arch, args.diffusion_steps, args.sampling_steps, args.class_cond, args.lr, args.batch_size * ngpus, args.class_cond_dropout
                )
        )

    elif ("celeba" in args.dataset or "fairface" in args.dataset) and args.class_cond:
        log_dir = os.path.join(
            args.save_dir,
            "female{}_male{}".format(args.female_ratio, args.male_ratio),
            "{}_diffusionstep_{}_samplestep_{}_condition_{}_lr_{}_bs_{}_dropprob_{}".format(
                args.arch, args.diffusion_steps, args.sampling_steps, args.class_cond, args.lr, args.batch_size * ngpus, args.class_cond_dropout
                )
        )
    elif "celeba" in args.dataset or "fairface" in args.dataset:
        log_dir = os.path.join(
            args.save_dir,
            "{}_diffusionstep_{}_samplestep_{}_condition_{}_lr_{}_bs_{}".format(
                args.arch, args.diffusion_steps, args.sampling_steps, args.class_cond, args.lr, args.batch_size * ngpus,
                )
        )
    os.makedirs(log_dir, exist_ok=True)

    # sampling
    if args.sampling_only:
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
        if "ema" in args.ckpt_name:
            os.makedirs(os.path.join(log_dir, "samples_ema"), exist_ok=True)
            np.savez(
                os.path.join(
                    log_dir,
                    "samples_ema",
                    f"{args.ckpt_name}_num{args.num_sampled_images}.npz",
                ),
                sampled_images,
                labels,
            )
        else:
            os.makedirs(os.path.join(log_dir, "samples"), exist_ok=True)
            np.savez(
                os.path.join(
                    log_dir,
                    "samples",
                    f"{args.ckpt_name}_num{args.num_sampled_images}.npz",
                ),
                sampled_images,
                labels,
            )
        if args.fix == "total" or args.fix == "color" or args.fix == "gray" or args.fix == "half":
            colorgray_dict = count_colorgray(sampled_images, threshold=args.threshold["pixel_val"])
            if args.local_rank == 0:
                print("Number of color: {} \n".format(colorgray_dict["num_color"]))
                print("Number of gray: {} \n".format(colorgray_dict["num_gray"]))
                print("Finish sampling from pretrained checkpoint! Return")
        return

    # Load dataset
    train_set = get_dataset(args.dataset, args.data_dir, metadata)
    sampler = DistributedSampler(train_set) if ngpus > 1 else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers = 4 * ngpus,
        pin_memory=True,
    )
    if args.local_rank == 0:
        print(
            f"Training dataset loaded: Number of batches: {len(train_loader)}, Number of images: {len(train_set)}"
        )
    
    logger = utils.logger(
        len(train_loader) * args.epochs, ["tb", "csv", "txt"], log_dir, args.ema_w
    )

    model_dir = os.path.join(log_dir, "ckpt")
    os.makedirs(model_dir, exist_ok=True)
    sample_dir = os.path.join(log_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    # ema model
    args.ema_dict = copy.deepcopy(model.state_dict())

    # lets start training the model
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        if args.dataset in ["celeba", "fairface", "fairface-base"]:
            train_one_epoch_gender(model, train_loader, diffusion, optimizer, logger, None, args, epoch)
        else:
            train_one_epoch(model, train_loader, diffusion, optimizer, logger, None, args, epoch)

        # sample during training
        if epoch > 0 and epoch % args.ckpt_sample_freq == 0:
            args.classifier_free_w = 0.0
            if ngpus > 1:
                sampled_images, _ = sample_N_images(
                        64,
                        model,
                        diffusion,
                        None,
                        args.sampling_steps,
                        args.batch_size,
                        metadata.num_channels,
                        metadata.image_size,
                        metadata.num_classes if args.class_cond else None,
                        args,
                    )
            else:
                sampled_images, _ = sample_N_images_nodist(
                        64,
                        model,
                        diffusion,
                        None,
                        args.sampling_steps,
                        args.batch_size,
                        metadata.num_channels,
                        metadata.image_size,
                        metadata.num_classes if args.class_cond else None,
                        args,
                    )
            if args.local_rank == 0:
                cv2.imwrite(
                    os.path.join(
                        sample_dir,
                        f"epoch_{epoch}.png",
                    ),
                    np.concatenate(sampled_images, axis=1)[:, :, ::-1],
                )

        if args.local_rank == 0 and epoch > 0 and epoch % args.ckpt_sample_freq == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    model_dir,
                    f"epoch_{epoch}.pth",
                ),
            )
            torch.save(
                args.ema_dict,
                os.path.join(
                    model_dir,
                    f"epoch_{epoch}_ema_{args.ema_w}.pth",
                ),
            )
    
    if ngpus > 1:
        sampled_images, _ = sample_N_images(
            64,
            model,
            diffusion,
            None,
            args.sampling_steps,
            args.batch_size,
            metadata.num_channels,
            metadata.image_size,
            metadata.num_classes if args.class_cond else None,
            args,
        )
    else:
        sampled_images, _ = sample_N_images_nodist(
            64,
            model,
            diffusion,
            None,
            args.sampling_steps,
            args.batch_size,
            metadata.num_channels,
            metadata.image_size,
            metadata.num_classes if args.class_cond else None,
            args,
        )
    if args.local_rank == 0:
        cv2.imwrite(
            os.path.join(
                sample_dir,
                "epoch_final.png",
            ),
            np.concatenate(sampled_images, axis=1)[:, :, ::-1],
        )

        torch.save(
            model.state_dict(),
            os.path.join(
                model_dir,
                "epoch_final.pth",
            ),
        )
        torch.save(
            args.ema_dict,
            os.path.join(
                model_dir,
                f"epoch_final_ema_{args.ema_w}.pth",
            ),
        )


if __name__ == "__main__":
    args = get_args()
    main(args)
