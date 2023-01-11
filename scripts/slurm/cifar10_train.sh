#!/bin/bash
#SBATCH --account=visualai    # Specify VisualAI

#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=8     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_2080:4          # the number of GPUs requested
#SBATCH --mem=48G             # memory 

#SBATCH --output=/n/fs/xl-diffbia/projects/minimal-diffusion/slurm_output/2023-01-11/cifar10_train_color0_gray1.txt            # where stdout and stderr will write to
#SBATCH -t 48:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=xl9353@cs.princeton.edu

source ~/.bashrc
conda init
conda activate diffusion-bias
cd /n/fs/xl-diffbia/projects/minimal-diffusion

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8105 main.py \
    --arch UNet --dataset cifar10 --epochs 1000 --batch-size 512 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 \
    --save-dir ./logs/ --date 2022-12-08 \
    --color 0.0 --grayscale 1.0