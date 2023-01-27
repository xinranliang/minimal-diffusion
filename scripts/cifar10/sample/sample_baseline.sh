#!/bin/bash
#SBATCH --account=visualai    # Specify VisualAI

#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=32     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_2080:8          # the number of GPUs requested
#SBATCH --mem=24G             # memory 

#SBATCH --output=/n/fs/xl-diffbia/projects/minimal-diffusion/slurm_output/2023-01-22/cifar10_sample_baseline.txt            # where stdout and stderr will write to
#SBATCH -t 48:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=xl9353@cs.princeton.edu

source ~/.bashrc
conda activate diffusion-bias
cd /n/fs/xl-diffbia/projects/minimal-diffusion


# uncond all color
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 4096 --num-sampled-images 50000 --sampling-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-01-22 \
    --color 1.0 --grayscale 0.0 \
    --ckpt-name cifar10_uncond_color1.0_gray0.0_epoch_1k \
    --pretrained-ckpt ./logs/2023-01-22/cifar10/color1.0_gray0.0/UNet_diffusionstep_1000_samplestep_1000_condition_False_lr_0.0001_bs_256/ckpt/epoch_1000.pth

# uncond all gray
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 4096 --num-sampled-images 50000 --sampling-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-01-22 \
    --color 0.0 --grayscale 1.0 \
    --ckpt-name cifar10_uncond_color0.0_gray1.0_epoch_1k \
    --pretrained-ckpt ./logs/2023-01-22/cifar10/color0.0_gray1.0/UNet_diffusionstep_1000_samplestep_1000_condition_False_lr_0.0001_bs_256/ckpt/epoch_1000.pth

# cond all color
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 4096 --num-sampled-images 50000 --sampling-only --sampling-steps 250 --class-cond \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-01-22 \
    --color 1.0 --grayscale 0.0 \
    --ckpt-name cifar10_cond_color1.0_gray0.0_epoch_1k \
    --pretrained-ckpt ./logs/2023-01-22/cifar10/color1.0_gray0.0/UNet_diffusionstep_1000_samplestep_1000_condition_True_lr_0.0001_bs_256/ckpt/epoch_1000.pth

# cond all gray
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 4096 --num-sampled-images 50000 --sampling-only --sampling-steps 250 --class-cond \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-01-22 \
    --color 0.0 --grayscale 1.0 \
    --ckpt-name cifar10_cond_color0.0_gray1.0_epoch_1k \
    --pretrained-ckpt ./logs/2023-01-22/cifar10/color0.0_gray1.0/UNet_diffusionstep_1000_samplestep_1000_condition_True_lr_0.0001_bs_256/ckpt/epoch_1000.pth