#!/bin/sh

#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=4     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_2080:1          # the number of GPUs requested
#SBATCH --mem=24G             # memory 
#SBATCH --output=/n/fs/xl-diffbia/projects/minimal-diffusion/slurm_output/2023-04-02/fixed_latent_viz.txt            # where stdout and stderr will write to
#SBATCH -t 12:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=xl9353@cs.princeton.edu

source ~/.bashrc
conda activate diffusion-bias
cd /n/fs/xl-diffbia/projects/minimal-diffusion

# CUDA_VISIBLE_DEVICES=0 python -m visualize.fixed_latent \
    # --arch UNet --dataset cifar10 --batch-size 500 --num-samples 500 --sampling-only --sampling-steps 250 --sampling-mode color --num-visualize 100 \
    # --data-dir ./datasets --save-dir ./logs/ --date 2023-04-01 --class-cond --classifier-free-w 0.0

# CUDA_VISIBLE_DEVICES=0 python -m visualize.fixed_latent \
    # --arch UNet --dataset mix-cifar10-imagenet --batch-size 500 --num-samples 500 --sampling-only --sampling-steps 250 --sampling-mode color --num-visualize 100 \
    # --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --class-cond --classifier-free-w 0.0

CUDA_VISIBLE_DEVICES=0 python -m visualize.fixed_latent \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 500 --num-samples 500 --sampling-only --sampling-steps 250 --sampling-mode gray --num-visualize 100 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --class-cond --classifier-free-w 0.0