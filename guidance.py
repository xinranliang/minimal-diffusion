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
from model.diffusion import GuassianDiffusion, sample_N_images
import model.unets as unets
from domain_classifier.cifar_imagenet import DomainClassifier as DC_cifar_imgnet
from domain_classifier.mnist_flip import DomainClassifier as DC_mnist, count_flip
import utils
from metrics.color_gray import count_colorgray, compute_colorgray
from main import get_args


def guidance_sample(args):
    # dataset
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
    
    # threshold for color/gray
    args.threshold = {}
    if args.dataset == "cifar10":
        args.threshold["norm_val"] = 0.006
        args.threshold["pixel_val"] = 0.74
    elif args.dataset == "mix-cifar10-imagenet":
        args.threshold["norm_val"] = 0.003
        args.threshold["pixel_val"] = 0.86
    
    if args.local_rank == 0:
        print("Checkpoint: {} \n".format(args.ckpt_name))
    # log dir
    str_lst = args.pretrained_ckpt.split("/")
    target_index = str_lst.index("ckpt")
    log_dir = os.path.join(*str_lst[:target_index])
    w_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0]

    for w in w_list:
        args.classifier_free_w = w

        # color/gray domain datasets: cifar10 or mix-cifar10-imagenet or cifar10-imagenet
        if "cifar10" in args.dataset:
            if args.sampling_only:
                # sample first time
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
            else:
                # otherwise load previous samples
                file_path = os.path.join(log_dir, "samples_ema", f"{args.ckpt_name}_num{args.num_sampled_images}_guidance{args.classifier_free_w}.npz",)
                file_load = np.load(file_path, allow_pickle=True)

                sampled_images = file_load['arr_0'] # shape = num_samples x height x width x n_channel
                labels = file_load['arr_1'] # empty if class_cond = False
                print("Loading samples and labels from {}".format(file_path))

            # overall
            colorgray_dict = count_colorgray(sampled_images, threshold=args.threshold["pixel_val"])
            colorgray_value = compute_colorgray(sampled_images, threshold=args.threshold["pixel_val"])
            # class-wise
            if args.class_cond:
                colorgray_dict_class = count_colorgray(sampled_images, args.threshold["pixel_val"], labels)
                colorgray_value_class = compute_colorgray(sampled_images, labels)

            if args.local_rank == 0:
                print("Sample with guidance {} \n".format(w))
                print("Number of color: {} \n".format(colorgray_dict["num_color"]))
                print("Number of gray: {} \n".format(colorgray_dict["num_gray"]))
                print("Mean value of channel std: {} \n".format(colorgray_value))

                print("Sample with guidance {} \n".format(w))
                print("Percentage of color by class: {} \n".format(colorgray_dict_class["ratio_color_classwise"]))
                print("Percentage of gray by class: {} \n".format(colorgray_dict_class["ratio_gray_classwise"]))
                print("Mean value of channel std by class: {} \n".format(colorgray_value_class))
            
            return
            
        # left/right horizontal flip domain dataset: mnist
        elif "mnist" in args.dataset:
            if args.sampling_only:
                # sample first time
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
            else:
                # otherwise load previous samples
                file_path = os.path.join(log_dir, "samples_ema", f"{args.ckpt_name}_num{args.num_sampled_images}_guidance{args.classifier_free_w}.npz",)
                file_load = np.load(file_path, allow_pickle=True)

                sampled_images = file_load['arr_0'] # shape = num_samples x height x width x n_channel
                labels = file_load['arr_1'] # empty if class_cond = False
                print("Loading samples and labels from {}".format(file_path))
                
            classifier = DC_mnist(
                num_classes = metadata.num_classes,
                num_domains = 2,
                learning_rate = 1e-3,
                weight_decay = 1e-4,
                device = args.device
            )
            if args.domain_classifier:
                classifier.load(args.domain_classifier)
                if args.local_rank == 0:
                    print("Loaded domain classifier checkpoint from {}.".format(args.domain_classifier))
            classifier.eval()

            # count number of left-flipped and right-flipped synthetic samples
            if args.local_rank == 0:
                print("Sample with guidance {} \n".format(w))

                transform_test = transforms.Compose(
                        [
                            # transforms.Grayscale(num_output_channels=1),
                            transforms.RandomResizedCrop(
                                28, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                            ),
                            transforms.ToTensor(),
                        ]
                )
                domain_dataset = ArrayToImageLabel(
                    samples = sampled_images,
                    labels = labels,
                    mode = "L",
                    transform = transform_test,
                    target_transform = None
                )
                domain_dataloader = DataLoader(
                    domain_dataset,
                    batch_size = args.batch_size,
                    shuffle = False,
                    num_workers = ngpus * 4
                )

                # overall number
                overall_counts = count_flip(domain_dataloader, classifier, classwise=False)
                print("Precent of regular synthetic digits over all classes: {:.3f}".format(overall_counts["num_left"] / args.num_sampled_images))
                print("Precent of flipped synthetic digits over all classes: {:.3f}".format(overall_counts["num_right"] / args.num_sampled_images))

                # group by classes
                classwise_counts = count_flip(domain_dataloader, classifier, classwise=True)
                assert args.num_sampled_images == np.sum(classwise_counts["num_samples"]), "number of total samples and sum of classwise counts do not match!"
                print("Precent of regular synthetic digits by classes: {}".format((classwise_counts["num_left"] / classwise_counts["num_samples"]).tolist()))
                print("Precent of flipped synthetic digits by classes: {}".format((classwise_counts["num_right"] / classwise_counts["num_samples"]).tolist()))
            
            return
            
        elif "cifar-superclass" in args.dataset:
            if args.sampling_only:
                # sample first time
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
                
            else:
                # otherwise load previous samples
                file_path = os.path.join(log_dir, "samples_ema", f"{args.ckpt_name}_num{args.num_sampled_images}_guidance{args.classifier_free_w}.npz",)
                file_load = np.load(file_path, allow_pickle=True)

                sampled_images = file_load['arr_0'] # shape = num_samples x height x width x n_channel
                labels = file_load['arr_1'] # empty if class_cond = False
                print("Loading samples and labels from {}".format(file_path))
            
            return
        
        elif args.dataset == "fairface":
            if args.sampling_only:
                # sample first time
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
            else:
                # otherwise load previous samples
                file_path = os.path.join(log_dir, "samples_ema", f"{args.ckpt_name}_num{args.num_sampled_images}_guidance{args.classifier_free_w}.npz",)
                file_load = np.load(file_path, allow_pickle=True)

                sampled_images = file_load['arr_0'] # shape = num_samples x height x width x n_channel
                labels = file_load['arr_1'] # empty if class_cond = False
                print("Loading samples and labels from {}".format(file_path))

                # domain classifier and count distribution
        
        elif args.dataset == "cifar-imagenet":
            if args.sampling_only:
                # sample first time
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
            else:
                # otherwise load previous samples
                file_path = os.path.join(log_dir, "samples_ema", f"{args.ckpt_name}_num{args.num_sampled_images}_guidance{args.classifier_free_w}.npz",)
                file_load = np.load(file_path, allow_pickle=True)

                sampled_images = file_load['arr_0'] # shape = num_samples x height x width x n_channel
                labels = file_load['arr_1'] # empty if class_cond = False
                print("Loading samples and labels from {}".format(file_path))

                # domain classifier and count distribution
        
        else:
            raise ValueError(f"Invalid dataset: {args.dataset}!")
            
        if args.sampling_only:
            if "ema" in args.ckpt_name:
                os.makedirs(os.path.join(log_dir, "samples_ema"), exist_ok=True)
                np.savez(
                    os.path.join(
                        log_dir,
                        "samples_ema",
                        f"{args.ckpt_name}_num{args.num_sampled_images}_guidance{args.classifier_free_w}.npz",
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
                        f"{args.ckpt_name}_num{args.num_sampled_images}_guidance{args.classifier_free_w}.npz",
                    ),
                    sampled_images,
                    labels,
                )
        
        return


if __name__ == "__main__":
    args = get_args()
    guidance_sample(args)