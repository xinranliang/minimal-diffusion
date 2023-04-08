#!/bin/sh

#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=8     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_2080:2          # the number of GPUs requested
#SBATCH --mem=24G             # memory 
#SBATCH --output=/n/fs/xl-diffbia/projects/minimal-diffusion/slurm_output/2023-04-07/cifar25k_imagenet0k_train_cond_droprob0.1.txt            # where stdout and stderr will write to
#SBATCH -t 60:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=xl9353@cs.princeton.edu

source ~/.bashrc
conda activate diffusion-bias
cd /n/fs/xl-diffbia/projects/minimal-diffusion

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port $3 main.py \
    --arch UNet --dataset cifar10-imagenet --epochs $2 --batch-size 256 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --ema_w 0.9995 --ckpt-sample-freq 500 \
    --class-cond --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-07 \
    --fix none --num-cifar10 25000 --num-imagenet $1

# number of cifar10 + imagenet samples to train model
    # 25k C + 0k I: 2500
    # 50k C + 0k I: 1250
    # 50k C + 25k I: 850
    # 50k C + 50k I: 625
    # 50k C + 75k I: 500
    # 50k C + 100k I: 425
    # 50k C + 125k I: 350
    # 50k C + 150k I: 325
    # 50k C + 175k I: 275
    # 50k C + 200k I: 250