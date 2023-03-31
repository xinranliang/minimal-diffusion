#!/bin/bash
#SBATCH --account=visualai    # Specify VisualAI

#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=16     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_2080:8          # the number of GPUs requested
#SBATCH --mem=24G             # memory 

#SBATCH --output=/n/fs/xl-diffbia/projects/minimal-diffusion/slurm_output/2023-03-19/cifar10_sample_colorgray_baseline_cond_puncond0.1.txt            # where stdout and stderr will write to
#SBATCH -t 24:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=xl9353@cs.princeton.edu

source ~/.bashrc
conda activate diffusion-bias
cd /n/fs/xl-diffbia/projects/minimal-diffusion


# cond all color
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 4096 --num-sampled-images 50000 --sampling-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-03-27 --color 1.0 --grayscale 0.0 --fix total \
    --class-cond --class-cond-dropout 0.1 --classifier-free-w 0.0 --diffusion-steps 1000 \
    --ckpt-name cifar10_cond_color1.0_gray0.0_puncond0.1_epoch1k_ema \
    --pretrained-ckpt ./logs/2023-03-27/cifar10/color1_gray0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1000_ema_0.9995.pth

# cond all gray
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 4096 --num-sampled-images 50000 --sampling-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-03-27 --color 0.0 --grayscale 1.0 --fix total \
    --class-cond --class-cond-dropout 0.1 --classifier-free-w 0.0 --diffusion-steps 1000 \
    --ckpt-name cifar10_cond_color0.0_gray1.0_puncond0.1_epoch1k_ema \
    --pretrained-ckpt ./logs/2023-03-27/cifar10/color0_gray1/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1000_ema_0.9995.pth
