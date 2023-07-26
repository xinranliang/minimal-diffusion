import os
import cv2
import copy
import math
import argparse
import numpy as np
from time import time
from tqdm import tqdm
from easydict import EasyDict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset.data import get_metadata, get_dataset
import model.unets
from utils import unsqueeze3x


class GuassianDiffusion:
    """Gaussian diffusion process with 1) Cosine schedule for beta values (https://arxiv.org/abs/2102.09672)
    2) L_simple training objective from https://arxiv.org/abs/2006.11239.
    """

    def __init__(self, timesteps=1000, device="cuda:0"):
        self.timesteps = timesteps
        self.device = device
        self.alpha_bar_scheduler = (
            lambda t: math.cos((t / self.timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
        )
        self.scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, self.timesteps, self.device
        )

        self.clamp_x0 = lambda x: x.clamp(-1, 1)
        self.get_x0_from_xt_eps = lambda xt, eps, t, scalars: (
            self.clamp_x0(
                1
                / unsqueeze3x(scalars.alpha_bar[t].sqrt())
                * (xt - unsqueeze3x((1 - scalars.alpha_bar[t]).sqrt()) * eps)
            )
        )
        self.get_pred_mean_from_x0_xt = (
            lambda xt, x0, t, scalars: unsqueeze3x(
                (scalars.alpha_bar[t].sqrt() * scalars.beta[t])
                / ((1 - scalars.alpha_bar[t]) * scalars.alpha[t].sqrt())
            )
            * x0
            + unsqueeze3x(
                (scalars.alpha[t] - scalars.alpha_bar[t])
                / ((1 - scalars.alpha_bar[t]) * scalars.alpha[t].sqrt())
            )
            * xt
        )

    def get_all_scalars(self, alpha_bar_scheduler, timesteps, device, betas=None):
        """
        Using alpha_bar_scheduler, get values of all scalars, such as beta, beta_hat, alpha, alpha_hat, etc.
        """
        all_scalars = {}
        if betas is None:
            all_scalars["beta"] = torch.from_numpy(
                np.array(
                    [
                        min(
                            1 - alpha_bar_scheduler(t + 1) / alpha_bar_scheduler(t),
                            0.999,
                        )
                        for t in range(timesteps)
                    ]
                )
            ).to(
                device
            )  # hardcoding beta_max to 0.999
        else:
            all_scalars["beta"] = betas
        all_scalars["beta_log"] = torch.log(all_scalars["beta"])
        all_scalars["alpha"] = 1 - all_scalars["beta"]
        all_scalars["alpha_bar"] = torch.cumprod(all_scalars["alpha"], dim=0)
        all_scalars["beta_tilde"] = (
            all_scalars["beta"][1:]
            * (1 - all_scalars["alpha_bar"][:-1])
            / (1 - all_scalars["alpha_bar"][1:])
        )
        all_scalars["beta_tilde"] = torch.cat(
            [all_scalars["beta_tilde"][0:1], all_scalars["beta_tilde"]]
        )
        all_scalars["beta_tilde_log"] = torch.log(all_scalars["beta_tilde"])
        return EasyDict(dict([(k, v.float()) for (k, v) in all_scalars.items()]))

    def sample_from_forward_process(self, x0, t):
        """Single step of the forward process, where we add noise in the image.
        Note that we will use this paritcular realization of noise vector (eps) in training.
        """
        eps = torch.randn_like(x0)
        xt = (
            unsqueeze3x(self.scalars.alpha_bar[t].sqrt()) * x0
            + unsqueeze3x((1 - self.scalars.alpha_bar[t]).sqrt()) * eps
        )
        return xt.float(), eps

    def sample_from_reverse_process(
        self, model, xT, timesteps=None, model_kwargs={}, ddim=False, w=0.0
    ):
        """Sampling images by iterating over all timesteps.

        model: diffusion model
        xT: Starting noise vector.
        timesteps: Number of sampling steps (can be smaller the default,
            i.e., timesteps in the diffusion process).
        model_kwargs: Additional kwargs for model (using it to feed class label for conditioning)
        ddim: Use ddim sampling (https://arxiv.org/abs/2010.02502). With very small number of
            sampling steps, use ddim sampling for better image quality.
        w: scale of classifier-free guidance, by default 0.0 is no guidance

        Return: An image tensor with identical shape as XT.
        """
        model.eval()
        final = xT

        # sub-sampling timesteps for faster sampling
        timesteps = timesteps or self.timesteps
        new_timesteps = np.linspace(
            0, self.timesteps - 1, num=timesteps, endpoint=True, dtype=int
        )
        alpha_bar = self.scalars["alpha_bar"][new_timesteps]
        new_betas = 1 - (
            alpha_bar / torch.nn.functional.pad(alpha_bar, [1, 0], value=1.0)[:-1]
        )
        scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, timesteps, self.device, new_betas
        )

        for i, t in zip(np.arange(timesteps)[::-1], new_timesteps[::-1]):
            with torch.no_grad():
                current_t = torch.tensor([t] * len(final), device=final.device)
                current_sub_t = torch.tensor([i] * len(final), device=final.device)
                pred_epsilon = model(final, current_t, **model_kwargs, mode="sample", w=w)
                # using xt+x0 to derive mu_t, instead of using xt+eps (former is more stable)
                pred_x0 = self.get_x0_from_xt_eps(
                    final, pred_epsilon, current_sub_t, scalars
                )
                pred_mean = self.get_pred_mean_from_x0_xt(
                    final, pred_x0, current_sub_t, scalars
                )
                if i == 0:
                    final = pred_mean
                else:
                    if ddim:
                        final = (
                            unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1]).sqrt()
                            * pred_x0
                            + (
                                1 - unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1])
                            ).sqrt()
                            * pred_epsilon
                        )
                    else:
                        final = pred_mean + unsqueeze3x(
                            scalars.beta_tilde[current_sub_t].sqrt()
                        ) * torch.randn_like(final)
                final = final.detach()
        return final



# training code
def train_one_epoch(
    model,
    dataloader,
    diffusion,
    optimizer,
    logger,
    lrs,
    args,
    epoch
):
    model.train()
    for step, (images, labels) in enumerate(dataloader):
        assert (images.max().item() <= 1) and (0 <= images.min().item())

        # must use [-1, 1] pixel range for images
        images, labels = (
            2 * images.to(args.device) - 1,
            labels.to(args.device) if args.class_cond else None,
        )
        t = torch.randint(diffusion.timesteps, (len(images),), dtype=torch.int64).to(
            args.device
        )
        xt, eps = diffusion.sample_from_forward_process(images, t)
        pred_eps = model(xt, t, y=labels, mode="train")

        loss = ((pred_eps - eps) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lrs is not None:
            lrs.step()

        # update ema_dict
        if args.local_rank == 0:
            new_dict = model.state_dict()
            for (k, v) in args.ema_dict.items():
                args.ema_dict[k] = (
                    args.ema_w * args.ema_dict[k] + (1 - args.ema_w) * new_dict[k]
                )
            logger.log(loss.item(), len(dataloader) * epoch + step)


def train_one_epoch_gender(
    model,
    dataloader,
    diffusion,
    optimizer,
    logger,
    lrs,
    args,
    epoch
):
    model.train()
    for step, (images, labels, domains) in enumerate(dataloader):
        assert (images.max().item() <= 1) and (0 <= images.min().item())

        # must use [-1, 1] pixel range for images
        images, labels = (
            2 * images.to(args.device) - 1,
            labels.to(args.device) if args.class_cond else None,
        )
        t = torch.randint(diffusion.timesteps, (len(images),), dtype=torch.int64).to(
            args.device
        )
        xt, eps = diffusion.sample_from_forward_process(images, t)
        pred_eps = model(xt, t, y=labels, mode="train")

        loss = ((pred_eps - eps) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lrs is not None:
            lrs.step()

        # update ema_dict
        if args.local_rank == 0:
            new_dict = model.state_dict()
            for (k, v) in args.ema_dict.items():
                args.ema_dict[k] = (
                    args.ema_w * args.ema_dict[k] + (1 - args.ema_w) * new_dict[k]
                )
            logger.log(loss.item(), len(dataloader) * epoch + step)


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
    with tqdm(total=math.ceil(N / (args.batch_size * num_processes))) as pbar:
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
            samples_list = [torch.zeros_like(gen_images) for _ in range(num_processes)]
            if args.class_cond:
                labels_list = [torch.zeros_like(y) for _ in range(num_processes)]
                dist.all_gather(labels_list, y, group)
                labels.append(torch.cat(labels_list).detach().cpu().numpy())

            dist.all_gather(samples_list, gen_images, group)
            samples.append(torch.cat(samples_list).detach().cpu().numpy())
            num_samples += len(xT) * num_processes
            pbar.update(1)
    samples = np.concatenate(samples).transpose(0, 2, 3, 1)[:N] # shape = num_samples x height x width x n_channel
    samples = (127.5 * (samples + 1)).astype(np.uint8)
    return (samples, np.concatenate(labels)[:N] if args.class_cond else None)


def sample_N_images_nodist(
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
    samples, labels, num_samples = [], [], 0
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
            if args.class_cond:
                labels.append(y.detach().cpu().numpy())

            samples.append(gen_images.detach().cpu().numpy())
            num_samples += len(xT)
            pbar.update(1)
    samples = np.concatenate(samples).transpose(0, 2, 3, 1)[:N] # shape = num_samples x height x width x n_channel
    samples = (127.5 * (samples + 1)).astype(np.uint8)
    return (samples, np.concatenate(labels)[:N] if args.class_cond else None)


def sample_color_images(
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

    samples, labels, num_samples = [], [], 0
    num_processes, group = dist.get_world_size(), dist.group.WORLD
    with tqdm(total=N) as pbar:
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
            samples_list = [torch.zeros_like(gen_images) for _ in range(num_processes)]
            dist.all_gather(samples_list, gen_images, group)
            curr_samples = torch.cat(samples_list).detach() # torch tensor: (num_samples x num_channel x height x width)
            channel_std = torch.std(curr_samples, dim=1).reshape(curr_samples.shape[0], curr_samples.shape[2] * curr_samples.shape[3]).mean(-1) # (num_samples x 1 x height x width)
            index_select = torch.nonzero(channel_std >= args.threshold["norm_val"]).reshape(-1)
            select_samples = torch.index_select(curr_samples, dim=0, index=index_select)
            samples.append(select_samples.cpu().numpy())

            if args.class_cond:
                labels_list = [torch.zeros_like(y) for _ in range(num_processes)]
                dist.all_gather(labels_list, y, group)
                curr_labels = torch.cat(labels_list).detach()
                select_labels = torch.index_select(curr_labels, dim=0, index=index_select)
                labels.append(select_labels.cpu().numpy())

            num_samples += index_select.shape[0]
            pbar.update(index_select.shape[0])
    samples = np.concatenate(samples).transpose(0, 2, 3, 1)[:N]
    samples = (127.5 * (samples + 1)).astype(np.uint8)
    return (samples, np.concatenate(labels)[:N] if args.class_cond else None)


def sample_gray_images(
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

    samples, labels, num_samples = [], [], 0
    num_processes, group = dist.get_world_size(), dist.group.WORLD
    with tqdm(total=N) as pbar:
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
            samples_list = [torch.zeros_like(gen_images) for _ in range(num_processes)]
            dist.all_gather(samples_list, gen_images, group)
            curr_samples = torch.cat(samples_list).detach() # torch tensor: (num_samples x num_channel x height x width)
            channel_std = torch.std(curr_samples, dim=1).reshape(curr_samples.shape[0], curr_samples.shape[2] * curr_samples.shape[3]).mean(-1) # (num_samples x 1 x height x width)
            index_select = torch.nonzero(channel_std < args.threshold["norm_val"]).reshape(-1)
            select_samples = torch.index_select(curr_samples, dim=0, index=index_select)
            samples.append(select_samples.cpu().numpy())

            if args.class_cond:
                labels_list = [torch.zeros_like(y) for _ in range(num_processes)]
                dist.all_gather(labels_list, y, group)
                curr_labels = torch.cat(labels_list).detach()
                select_labels = torch.index_select(curr_labels, dim=0, index=index_select)
                labels.append(select_labels.cpu().numpy())

            num_samples += index_select.shape[0]
            pbar.update(index_select.shape[0])
    samples = np.concatenate(samples).transpose(0, 2, 3, 1)[:N]
    samples = (127.5 * (samples + 1)).astype(np.uint8)
    return (samples, np.concatenate(labels)[:N] if args.class_cond else None)


def sample_N_images_cond(
    N,
    model,
    diffusion,
    class_label,
    xT=None,
    sampling_steps=250,
    batch_size=64,
    num_channels=3,
    image_size=32,
    args=None,
):
    # sample n images from each class for total k classes
    """use this function to sample any number of images from a given
        diffusion model and diffusion process.

    Args:
        N : Number of images for one class
        model : Diffusion model
        diffusion : Diffusion process
        xT : Starting instantiation of noise vector.
        sampling_steps : Number of sampling steps.
        batch_size : Batch-size for sampling.
        num_channels : Number of channels in the image.
        image_size : Image size (assuming square images).
        num_classes : Number of classes in the dataset, assume class-conditional model
        args : All args from the argparser.

    Returns: Numpy array (num_classes x n x image_size) with images and corresponding labels.
    """
    # assert class-conditional
    assert args.class_cond 

    samples, labels, num_samples = [], [], 0
    num_processes, group = dist.get_world_size(), dist.group.WORLD

    with tqdm(total=math.ceil(N / (args.batch_size * num_processes))) as pbar:
        while num_samples < N:
            # initialize noise
            if xT is None:
                xT = (
                        torch.randn(batch_size, num_channels, image_size, image_size)
                        .float()
                        .to(args.device)
                    )
                # specify class label
            y = torch.zeros(size=(len(xT),), dtype=torch.int64).to(args.device)
            y.fill_(class_label)

            gen_images = diffusion.sample_from_reverse_process(
                model, xT, sampling_steps, {"y": y}, args.ddim, args.classifier_free_w
            )
            samples_list = [torch.zeros_like(gen_images) for _ in range(num_processes)]

            labels_list = [torch.zeros_like(y) for _ in range(num_processes)]
            dist.all_gather(labels_list, y, group)
            labels.append(torch.cat(labels_list).detach().cpu().numpy())

            dist.all_gather(samples_list, gen_images, group)
            samples.append(torch.cat(samples_list).detach().cpu().numpy())
            num_samples += len(xT) * num_processes
            pbar.update(1)
            
    samples = np.concatenate(samples).transpose(0, 2, 3, 1)[:N] # shape = num_samples x height x width x n_channel
    samples = (127.5 * (samples + 1)).astype(np.uint8)
    return (samples, np.concatenate(labels)[:N])


def sample_N_images_classifier(
    N,
    model,
    diffusion,
    classifer,
    transform_test,
    xT=None,
    sampling_steps=250,
    batch_size=64,
    num_channels=3,
    image_size=32,
    num_classes=None,
    args=None,
):
    samples, labels, num_samples = [], [], 0
    num_processes, group = dist.get_world_size(), dist.group.WORLD
    with tqdm(total=N) as pbar:
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
            samples_list = [torch.zeros_like(gen_images) for _ in range(num_processes)]
            if args.class_cond:
                labels_list = [torch.zeros_like(y) for _ in range(num_processes)]
                dist.all_gather(labels_list, y, group)
                curr_labels = torch.cat(labels_list).detach()
                # labels.append(torch.cat(labels_list).detach().cpu().numpy())

            dist.all_gather(samples_list, gen_images, group)
            curr_samples = torch.cat(samples_list).detach()
            # samples.append(torch.cat(samples_list).detach().cpu().numpy())

            curr_input = (curr_samples + 1.0) / 2 # normalize to 0-1
            curr_input = transform_test(curr_input) # normalize by mean std
            curr_pred = classifer.predict(curr_input, adapt=True)
            index_select = torch.nonzero(curr_pred == 0).squeeze()
            select_samples = torch.index_select(curr_samples, dim=0, index=index_select)
            samples.append(select_samples.cpu().numpy())
            if args.class_cond:
                select_labels = torch.index_select(curr_labels, dim=0, index=index_select)
                labels.append(select_labels.cpu().numpy())

            num_samples += index_select.shape[0]
            pbar.update(index_select.shape[0])
    samples = np.concatenate(samples).transpose(0, 2, 3, 1)[:N] # shape = num_samples x height x width x n_channel
    samples = (127.5 * (samples + 1)).astype(np.uint8)
    return (samples, np.concatenate(labels)[:N] if args.class_cond else None)