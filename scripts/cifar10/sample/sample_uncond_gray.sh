#!/bin/bash
#SBATCH --account=visualai    # Specify VisualAI

#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=32     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_2080:8          # the number of GPUs requested
#SBATCH --mem=24G             # memory 

#SBATCH --output=/n/fs/xl-diffbia/projects/minimal-diffusion/slurm_output/2023-01-22/cifar10_uncond_gray.txt            # where stdout and stderr will write to
#SBATCH -t 48:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=xl9353@cs.princeton.edu

source ~/.bashrc
conda activate diffusion-bias
cd /n/fs/xl-diffbia/projects/minimal-diffusion

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 4096 --num-sampled-images 50000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-01-22 --color 0.01 --grayscale 0.99 \
    --ckpt-name cifar10_uncond_color0.01_gray0.99_epoch1000_ema \
    --pretrained-ckpt ./logs/2023-01-22/cifar10/color0.01_gray0.99/UNet_diffusionstep_1000_samplestep_1000_condition_False_lr_0.0001_bs_256/ckpt/epoch_1000_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 4096 --num-sampled-images 50000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-01-22 --color 0.05 --grayscale 0.95 \
    --ckpt-name cifar10_uncond_color0.05_gray0.95_epoch1000_ema \
    --pretrained-ckpt ./logs/2023-01-22/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_1000_condition_False_lr_0.0001_bs_256/ckpt/epoch_1000_ema_0.9995.pth 

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 4096 --num-sampled-images 50000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-01-22 --color 0.1 --grayscale 0.9 \
    --ckpt-name cifar10_uncond_color0.1_gray0.9_epoch1000_ema \
    --pretrained-ckpt ./logs/2023-01-22/cifar10/color0.1_gray0.9/UNet_diffusionstep_1000_samplestep_1000_condition_False_lr_0.0001_bs_256/ckpt/epoch_1000_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 4096 --num-sampled-images 50000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-01-22 --color 0.9 --grayscale 0.1 \
    --ckpt-name cifar10_uncond_color0.9_gray0.1_epoch1000_ema \
    --pretrained-ckpt ./logs/2023-01-22/cifar10/color0.9_gray0.1/UNet_diffusionstep_1000_samplestep_1000_condition_False_lr_0.0001_bs_256/ckpt/epoch_1000_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 4096 --num-sampled-images 50000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-01-22 --color 0.95 --grayscale 0.05 \
    --ckpt-name cifar10_uncond_color0.95_gray0.05_epoch1000_ema \
    --pretrained-ckpt ./logs/2023-01-22/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_1000_condition_False_lr_0.0001_bs_256/ckpt/epoch_1000_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 4096 --num-sampled-images 50000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-01-22 --color 0.99 --grayscale 0.01 \
    --ckpt-name cifar10_uncond_color0.99_gray0.01_epoch1000_ema \
    --pretrained-ckpt ./logs/2023-01-22/cifar10/color0.01_gray0.99/UNet_diffusionstep_1000_samplestep_1000_condition_False_lr_0.0001_bs_256/ckpt/epoch_1000_ema_0.9995.pth
