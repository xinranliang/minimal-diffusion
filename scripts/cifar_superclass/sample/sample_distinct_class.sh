#!/bin/sh

#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=32     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_2080:8          # the number of GPUs requested
#SBATCH --mem=24G             # memory 
#SBATCH --output=/n/fs/xl-diffbia/projects/minimal-diffusion/slurm_output/2023-06-25/cifar_distinct_class_sample_guidance.txt            # where stdout and stderr will write to
#SBATCH -t 48:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=xl9353@cs.princeton.edu

source ~/.bashrc
conda activate diffusion-bias
cd /n/fs/xl-diffbia/projects/minimal-diffusion

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar-superclass --batch-size 5000 --num-sampled-images 50000 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --lr 1e-4 --class-cond --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-06-25 --guidance \
    --pretrained-ckpt ./logs/2023-06-25/cifar-superclass/group_distinct/front0.1_back0.9/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --semantic-group distinct --front-ratio 0.1 --back-ratio 0.9

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar-superclass --batch-size 5000 --num-sampled-images 50000 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --lr 1e-4 --class-cond --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-06-25 --guidance \
    --pretrained-ckpt ./logs/2023-06-25/cifar-superclass/group_distinct/front0.3_back0.7/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --semantic-group distinct --front-ratio 0.3 --back-ratio 0.7

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar-superclass --batch-size 5000 --num-sampled-images 50000 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --lr 1e-4 --class-cond --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-06-25 --guidance \
    --pretrained-ckpt ./logs/2023-06-25/cifar-superclass/group_distinct/front0.5_back0.5/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --semantic-group distinct --front-ratio 0.5 --back-ratio 0.5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar-superclass --batch-size 5000 --num-sampled-images 50000 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --lr 1e-4 --class-cond --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-06-25 --guidance \
    --pretrained-ckpt ./logs/2023-06-25/cifar-superclass/group_distinct/front0.7_back0.3/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --semantic-group distinct --front-ratio 0.7 --back-ratio 0.3

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar-superclass --batch-size 5000 --num-sampled-images 50000 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --lr 1e-4 --class-cond --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-06-25 --guidance \
    --pretrained-ckpt ./logs/2023-06-25/cifar-superclass/group_distinct/front0.9_back0.1/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --semantic-group distinct --front-ratio 0.9 --back-ratio 0.1
