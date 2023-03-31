#!/bin/bash
#SBATCH --account=visualai    # Specify VisualAI

#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=32     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_3090:8          # the number of GPUs requested
#SBATCH --mem=32G             # memory 

#SBATCH --output=/n/fs/xl-diffbia/projects/minimal-diffusion/slurm_output/2023-03-27/cifar10_guidance_fixtotal_cond_dropprob0.1_count.txt            # where stdout and stderr will write to
#SBATCH -t 24:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=xl9353@cs.princeton.edu

source ~/.bashrc
conda activate diffusion-bias
cd /n/fs/xl-diffbia/projects/minimal-diffusion


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 10000 --num-sampled-images 50000 --guidance --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-03-27 --color 0.95 --grayscale 0.05 --fix total \
    --class-cond --class-cond-dropout 0.1 --classifier-free-w 0.0 --diffusion-steps 1000 \
    --ckpt-name cifar10_cond_color0.95_gray0.05_puncond0.1_epoch1k_ema \
    --pretrained-ckpt ./logs/2023-03-27/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1000_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 10000 --num-sampled-images 50000 --guidance --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-03-27 --color 0.9 --grayscale 0.1 --fix total \
    --class-cond --class-cond-dropout 0.1 --classifier-free-w 0.0 --diffusion-steps 1000 \
    --ckpt-name cifar10_cond_color0.9_gray0.1_puncond0.1_epoch1k_ema \
    --pretrained-ckpt ./logs/2023-03-27/cifar10/color0.9_gray0.1/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1000_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 10000 --num-sampled-images 50000 --guidance --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-03-27 --color 0.7 --grayscale 0.3 --fix total \
    --class-cond --class-cond-dropout 0.1 --classifier-free-w 0.0 --diffusion-steps 1000 \
    --ckpt-name cifar10_cond_color0.7_gray0.3_puncond0.1_epoch1k_ema \
    --pretrained-ckpt ./logs/2023-03-27/cifar10/color0.7_gray0.3/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1000_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 10000 --num-sampled-images 50000 --guidance --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-03-27 --color 0.5 --grayscale 0.5 --fix total \
    --class-cond --class-cond-dropout 0.1 --classifier-free-w 0.0 --diffusion-steps 1000 \
    --ckpt-name cifar10_cond_color0.5_gray0.5_puncond0.1_epoch1k_ema \
    --pretrained-ckpt ./logs/2023-03-27/cifar10/color0.5_gray0.5/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1000_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 10000 --num-sampled-images 50000 --guidance --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-03-27 --color 0.3 --grayscale 0.7 --fix total \
    --class-cond --class-cond-dropout 0.1 --classifier-free-w 0.0 --diffusion-steps 1000 \
    --ckpt-name cifar10_cond_color0.3_gray0.7_puncond0.1_epoch1k_ema \
    --pretrained-ckpt ./logs/2023-03-27/cifar10/color0.3_gray0.7/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1000_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 10000 --num-sampled-images 50000 --guidance --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-03-27 --color 0.1 --grayscale 0.9 --fix total \
    --class-cond --class-cond-dropout 0.1 --classifier-free-w 0.0 --diffusion-steps 1000 \
    --ckpt-name cifar10_cond_color0.1_gray0.9_puncond0.1_epoch1k_ema \
    --pretrained-ckpt ./logs/2023-03-27/cifar10/color0.1_gray0.9/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1000_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 10000 --num-sampled-images 50000 --guidance --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-03-27 --color 0.05 --grayscale 0.95 --fix total \
    --class-cond --class-cond-dropout 0.1 --classifier-free-w 0.0 --diffusion-steps 1000 \
    --ckpt-name cifar10_cond_color0.05_gray0.95_puncond0.1_epoch1k_ema \
    --pretrained-ckpt ./logs/2023-03-27/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1000_ema_0.9995.pth
