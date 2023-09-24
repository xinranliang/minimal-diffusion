import os
import cv2
import copy
import math
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from easydict import EasyDict

import torch
from torchvision import datasets, transforms
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset.data import get_metadata, get_domain_dataset
from dataset.utils import ArrayToImageLabel

from model.diffusion import GuassianDiffusion, sample_N_images
import model.unets as unets

from domain_classifier.cifar_imagenet import DomainClassifier as DC_cifar_imgnet, count_cifar_imgnet, eval_cifar_imgnet_domain
from domain_classifier.mnist_flip import DomainClassifier as DC_mnist, count_flip
from domain_classifier.fairface import DomainClassifier as DC_fairface, count_fairface, count_fairface_real
from domain_classifier.celeba import DomainClassifier as DC_celeba
from metrics.color_gray import count_colorgray, compute_colorgray

import utils
from main import get_args

from plot.plot_cifar_imgnet import plot_cifar_imgnet_hist, plot_check_domaineval
from plot.plot_fairface_gender import plot_fairface_hist


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

    # first predict real samples distribution
    if not args.sampling_only:
        if "mnist" in args.dataset:
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

            real_dataset = get_domain_dataset(args.dataset, args.data_dir, metadata)
            real_dataloader = DataLoader(
                real_dataset,
                batch_size = args.batch_size,
                shuffle = False,
                num_workers = ngpus * 4
            )
            real_overall_counts = count_flip(real_dataloader, classifier, classwise=False)
            print("Precent of predicted regular real digits over all classes: {:.5f}".format(overall_counts["num_left"] / len(real_dataset)))
            print("Precent of predicted flipped real digits over all classes: {:.5f}".format(overall_counts["num_right"] / len(real_dataset)))
        
        elif args.dataset == "cifar-imagenet":
            # domain classifier and count distribution
            classifier = DC_cifar_imgnet(
                num_classes = 2,
                arch = "resnet50",
                pretrained = False,
                learning_rate = 0.001,
                weight_decay = 0.0001,
                device = args.device
            )
            if args.domain_classifier:
                classifier.load(args.domain_classifier)
                if args.local_rank == 0:
                    print("Loaded domain classifier checkpoint from {}.".format(args.domain_classifier))
            classifier.eval()

            real_dataset = get_domain_dataset(args.dataset, args.data_dir, metadata)
            real_dataloader = DataLoader(
                real_dataset,
                batch_size = args.batch_size,
                shuffle = False,
                num_workers = 4
            )

            real_count_dict = count_cifar_imgnet(real_dataloader, classifier, return_type=["percent", "histogram"])

            # print result
            print("Percent of predicted real CIFAR samples: {:.5f}".format(real_count_dict["percent"]["num_cifar"] / len(real_dataset)))
            print("Percent of predicted real ImageNet samples: {:.5f}".format(real_count_dict["percent"]["num_imgnet"] / len(real_dataset)))
        
        elif args.dataset == "cifar-imagenet-check":
            # domain classifier and count distribution
            classifier = DC_cifar_imgnet(
                num_classes = 2,
                arch = "resnet50",
                pretrained = False,
                learning_rate = 0.001,
                weight_decay = 0.0001,
                device = args.device
            )
            if args.domain_classifier:
                classifier.load(args.domain_classifier)
                if args.local_rank == 0:
                    print("Loaded domain classifier checkpoint from {}.".format(args.domain_classifier))
            classifier.eval()

        elif args.dataset == "fairface":
            classifier = DC_fairface(
                num_classes = 2,
                arch = "resnet50",
                pretrained = True,
                learning_rate = 0.0001,
                weight_decay = 0.00001,
                device = args.device
            )
            if args.domain_classifier:
                classifier.load(args.domain_classifier)
                if args.local_rank == 0:
                    print("Loaded domain classifier checkpoint from {}.".format(args.domain_classifier))
            classifier.eval()

            real_dataset = get_domain_dataset(args.dataset, args.data_dir, metadata)
            real_dataloader = DataLoader(
                real_dataset,
                batch_size = args.batch_size,
                shuffle = False,
                num_workers = 4
            )

            real_count_dict = count_fairface_real(real_dataloader, classifier, return_type=["percent", "histogram"])
            # print result
            print("Percent of predicted real female samples: {:.5f}".format(real_count_dict["percent"]["num_female"] / len(real_dataset)))
            print("Percent of predicted real male samples: {:.5f}".format(real_count_dict["percent"]["num_male"] / len(real_dataset)))

    # classifier-free guidance, for synthetic samples
    if "cifar10" in args.dataset or "mnist" in args.dataset:
        w_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0]
    elif "fairface" in args.dataset or "cifar-imagenet" in args.dataset:
        w_list = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    # for plotting
    if not args.sampling_only:
        prob_hist_list = []

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
                
            assert classifier is not None, f"Invalid domain classifier for {args.dataset} dataset"

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
                print("Precent of regular synthetic digits over all classes: {:.5f}".format(overall_counts["num_left"] / args.num_sampled_images))
                print("Precent of flipped synthetic digits over all classes: {:.5f}".format(overall_counts["num_right"] / args.num_sampled_images))

                # group by classes
                classwise_counts = count_flip(domain_dataloader, classifier, classwise=True)
                assert args.num_sampled_images == np.sum(classwise_counts["num_samples"]), "number of total samples and sum of classwise counts do not match!"
                print("Precent of regular synthetic digits by classes: {}".format((classwise_counts["num_left"] / classwise_counts["num_samples"]).tolist()))
                print("Precent of flipped synthetic digits by classes: {}".format((classwise_counts["num_right"] / classwise_counts["num_samples"]).tolist()))
            
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
                if args.local_rank == 0:
                    transform_test = transforms.Compose(
                        [
                            transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalized by imagenet mean std
                        ]
                    )
                    domain_dataset = ArrayToImageLabel(
                        samples = sampled_images,
                        labels = None,
                        mode = "RGB",
                        transform = transform_test,
                        target_transform = None
                    )
                    domain_dataloader = DataLoader(
                        domain_dataset,
                        batch_size = args.batch_size,
                        shuffle = False,
                        num_workers = ngpus * 4
                    )

                    count_dict = count_fairface(domain_dataloader, classifier, return_type=["percent", "histogram"])
                    # print result
                    print("Percent of female samples: {:.5f}".format(count_dict["percent"]["num_female"] / args.num_sampled_images))
                    print("Percent of male samples: {:.5f}".format(count_dict["percent"]["num_male"] / args.num_sampled_images))
                    prob_hist_list.append(count_dict["histogram"])

                    # subsample for more results
                    if args.class_cond:
                        assert sampled_images.shape[0] == labels.shape[0], "Shape of images and lables does not match"
                    sub_results = []
                    for _ in range(5):
                        args.seed = (
                            os.getpid()
                            + int(datetime.now().strftime("%S%f"))
                            + int.from_bytes(os.urandom(2), "big")
                        )
                        torch.manual_seed(args.seed + args.local_rank)
                        np.random.seed(args.seed + args.local_rank)
                        if args.local_rank == 0:
                            print("Subsample with random seed {}".format(args.seed + args.local_rank))
                        
                        select_indices = np.random.choice(args.num_sampled_images, 10000, replace=False)
                        sub_sampled_images = np.take(sampled_images, select_indices, axis=0)
                        sub_labels = np.take(labels, select_indices, axis=0)
                        domain_dataset = ArrayToImageLabel(
                            samples = sub_sampled_images,
                            labels = None,
                            mode = "RGB",
                            transform = transform_test,
                            target_transform = None
                        )
                        domain_dataloader = DataLoader(
                            domain_dataset,
                            batch_size = args.batch_size,
                            shuffle = False,
                            num_workers = ngpus * 4
                        )

                        count_dict = count_fairface(domain_dataloader, classifier, return_type=["percent", "histogram"])
                        # print result
                        sub_results.append(np.round(count_dict["percent"]["num_female"] / 10000, 5))
                    print(f"Percent of female samples: {sub_results}")
            
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

                if args.local_rank == 0:
                    transform_test = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])
                        ]
                    )
                    domain_dataset = ArrayToImageLabel(
                        samples = sampled_images,
                        labels = None,
                        mode = "RGB",
                        transform = transform_test,
                        target_transform = None
                    )
                    domain_dataloader = DataLoader(
                        domain_dataset,
                        batch_size = args.batch_size,
                        shuffle = False,
                        num_workers = ngpus * 4
                    )
                    count_dict = count_cifar_imgnet(domain_dataloader, classifier, return_type=["percent", "histogram"])

                    # print result
                    print("Percent of CIFAR samples: {:.5f}".format(count_dict["percent"]["num_cifar"] / args.num_sampled_images))
                    print("Percent of ImageNet samples: {:.5f}".format(count_dict["percent"]["num_imgnet"] / args.num_sampled_images))

                    prob_hist_list.append(count_dict["histogram"])

                    # subsample for more results
                    if args.class_cond:
                        assert sampled_images.shape[0] == labels.shape[0], "Shape of images and lables does not match"
                    sub_results = []
                    for _ in range(5):
                        args.seed = (
                            os.getpid()
                            + int(datetime.now().strftime("%S%f"))
                            + int.from_bytes(os.urandom(2), "big")
                        )
                        torch.manual_seed(args.seed + args.local_rank)
                        np.random.seed(args.seed + args.local_rank)
                        if args.local_rank == 0:
                            print("Subsample with random seed {}".format(args.seed + args.local_rank))
                        
                        select_indices = np.random.choice(args.num_sampled_images, 10000, replace=False)
                        sub_sampled_images = np.take(sampled_images, select_indices, axis=0)
                        sub_labels = np.take(labels, select_indices, axis=0)
                        domain_dataset = ArrayToImageLabel(
                            samples = sub_sampled_images,
                            labels = None,
                            mode = "RGB",
                            transform = transform_test,
                            target_transform = None
                        )
                        domain_dataloader = DataLoader(
                            domain_dataset,
                            batch_size = args.batch_size,
                            shuffle = False,
                            num_workers = ngpus * 4
                        )

                        count_dict = count_cifar_imgnet(domain_dataloader, classifier, return_type=["percent", "histogram"])
                        # print result
                        sub_results.append(np.round(count_dict["percent"]["num_cifar"] / 10000, 5))
                    print(f"Percent of CIFAR samples: {sub_results}")
            
        elif args.dataset == "cifar-imagenet-check":
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

                if args.local_rank == 0:
                    transform_test = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])
                        ]
                    )
                    domain_dataset = ArrayToImageLabel(
                        samples = sampled_images,
                        labels = labels, # should be 20-class labels
                        mode = "RGB",
                        transform = transform_test,
                        target_transform = None
                    )
                    domain_dataloader = DataLoader(
                        domain_dataset,
                        batch_size = args.batch_size,
                        shuffle = False,
                        num_workers = ngpus * 4
                    )

                    return_dict = eval_cifar_imgnet_domain(domain_dataloader, classifier, return_type = ["true", "pred", "full"])

                    save_folder = os.path.join(log_dir, "figures")
                    os.makedirs(save_folder, exist_ok=True)
                    plot_check_domaineval(return_dict, save_folder, args.classifier_free_w, [args.num_cifar, args.num_imagenet])
                    
        else:
            raise ValueError(f"Invalid dataset: {args.dataset}!")
            
        if args.sampling_only:
            print(f"sampling with guidance scale {args.classifier_free_w}")
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
    
    # have additional results, plot and save
    if not args.sampling_only and len(prob_hist_list) > 0:
        os.makedirs(os.path.join(log_dir, "figures"), exist_ok=True)
        if args.dataset == "cifar-imagenet":
            plot_cifar_imgnet_hist(real_count_dict["histogram"], prob_hist_list, os.path.join(log_dir, "figures"))
        elif args.dataset == "fairface":
            plot_fairface_hist(real_count_dict["histogram"], prob_hist_list, os.path.join(log_dir, "figures"))
        
    return


if __name__ == "__main__":
    args = get_args()
    guidance_sample(args)