#!/bin/bash
#SBATCH --account=visualai    # Specify VisualAI

#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=8     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_3090:4          # the number of GPUs requested
#SBATCH --mem=48G             # memory 

#SBATCH --output=/n/fs/xl-diffbia/projects/minimal-diffusion/output.txt            # where stdout and stderr will write to
#SBATCH -t 24:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=xl9353@cs.princeton.edu

source ~/.bashrc
conda init
conda activate diffusion-bias
cd /n/fs/xl-diffbia/projects/minimal-diffusion

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 512 --num-sampled-images 50000 --sampling-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2022-12-08 \
    --color 0.01 --grayscale 0.99 \
    --ckpt-name cifar10_color0.01_gray0.99_epoch_950 \
    --pretrained-ckpt ./logs/2022-12-08/cifar10/color0.01_gray0.99/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/ckpt/epoch_950.pth 

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 512 --num-sampled-images 50000 --sampling-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2022-12-08 \
    --color 0.05 --grayscale 0.95 \
    --ckpt-name cifar10_color0.05_gray0.95_epoch_950 \
    --pretrained-ckpt ./logs/2022-12-08/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/ckpt/epoch_950.pth 

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 512 --num-sampled-images 50000 --sampling-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2022-12-08 \
    --color 0.95 --grayscale 0.05 \
    --ckpt-name cifar10_color0.95_gray0.05_epoch_950 \
    --pretrained-ckpt ./logs/2022-12-08/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/ckpt/epoch_950.pth 

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 512 --num-sampled-images 50000 --sampling-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2022-12-08 \
    --color 0.99 --grayscale 0.01 \
    --ckpt-name cifar10_color0.99_gray0.01_epoch_950 \
    --pretrained-ckpt ./logs/2022-12-08/cifar10/color0.99_gray0.01/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/ckpt/epoch_950.pth 