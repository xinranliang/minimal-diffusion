#!/bin/bash
#SBATCH --account=visualai    # Specify VisualAI

#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=16     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_2080:8          # the number of GPUs requested
#SBATCH --mem=48G             # memory 

#SBATCH --output=/n/fs/xl-diffbia/projects/minimal-diffusion/slurm_output/2023-01-14/cifar10_sample_baseline.txt            # where stdout and stderr will write to
#SBATCH -t 4:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=xl9353@cs.princeton.edu

source ~/.bashrc
conda activate diffusion-bias
cd /n/fs/xl-diffbia/projects/minimal-diffusion

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 4096 --num-sampled-images 50000 --sampling-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2022-12-08 \
    --color 1.0 --grayscale 0.0 \
    --ckpt-name cifar10_color1_gray0_epoch_1000 \
    --pretrained-ckpt ./logs/2022-12-08/cifar10/color1.0_gray0.0/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/ckpt/epoch_final.pth

"""CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 4096 --num-sampled-images 50000 --sampling-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2022-12-08 \
    --color 1.0 --grayscale 0.0 \
    --ckpt-name cifar10_color1_gray0_epoch_1000_ema \
    --pretrained-ckpt ./logs/2022-12-08/cifar10/color1.0_gray0.0/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/ckpt/epoch_final_ema_0.9995.pth""" 

