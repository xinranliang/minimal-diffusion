#!/bin/sh

#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=8     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_2080:2          # the number of GPUs requested
#SBATCH --mem=24G             # memory 
#SBATCH --output=/n/fs/xl-diffbia/projects/minimal-diffusion/slurm_output/2023-04-03/mix_cifar10_imagenet_train_cond_color15_gray22.5_droprob0.1.txt            # where stdout and stderr will write to
#SBATCH -t 60:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=xl9353@cs.princeton.edu

source ~/.bashrc
conda activate diffusion-bias
cd /n/fs/xl-diffbia/projects/minimal-diffusion


CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port $3 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --epochs $2 --batch-size 256 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --ema_w 0.9995 --ckpt-sample-freq 300 \
    --class-cond --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-03 \
    --color 15000 --grayscale $1 --fix color


# fix color 15k
    # 15k C + 0 G: 4250
    # 15k C + 7.5k G: 2850
    # 15k C + 15k G: 2125
    # 15k C + 22.5k G: 1700
    # 15k C + 30k G: 1420
    # 15k C + 60k G: 850
    # 15k C + 90k G: 610
    # 15k C + 120k G: 475
    # 15k C + 150k G: 390
    # 15k C + 180k G: 330
    # 15k C + 210k G: 285
    # 15k C + 240k G: 250