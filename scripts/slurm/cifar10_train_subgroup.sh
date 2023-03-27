#!/bin/sh

#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=8     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_2080:2          # the number of GPUs requested
#SBATCH --mem=24G             # memory 
#SBATCH --output=/n/fs/xl-diffbia/projects/minimal-diffusion/slurm_output/2023-03-27/cifar10_train_cond_color47.5_gray0_droprob0.1.txt            # where stdout and stderr will write to
#SBATCH -t 48:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=xl9353@cs.princeton.edu

source ~/.bashrc
conda activate diffusion-bias
cd /n/fs/xl-diffbia/projects/minimal-diffusion

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port $3 main.py \
    --arch UNet --dataset cifar10 --epochs $2 --batch-size 256 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --ema_w 0.9995 --ckpt-sample-freq 250 \
    --class-cond --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-03-27 \
    --color $1 --grayscale 0 --fix subgroup

# compute number of training epochs:
    # 2500C + 0G: 20001
    # 2500C + 500G: 16001
    # 2500C + 1000G: 14001
    # 2500C + 1500G: 13001
    # 2500C + 2000G: 11001
    # 2500C + 2500G: 10001
    # 2500C + 5000G: 7001
    # 2500C + 10000G: 4001
    # 2500C + 20000G: 2201
    # 2500C + 30000G: 1501
    # 2500C + 40000G: 1201

    # 5000C + 0G: 10001
    # 5000C + 5000G: 5001
    # 5000C + 10000G: 4001
    # 5000C + 20000G: 2001
    # 5000C + 30000G: 1501
    # 5000C + 40000G: 1001

    # 25k C + 0k G: 2001
    # 25k C + 5k G: 1801
    # 25k C + 10k G: 1601
    # 25k C + 15k G: 1401
    # 25k C + 20k G: 1201
    # 25k C + 25k G: 1001

    # 15k C + 0k G: 3334
    # 15k C + 5k G: 2500
    # 15k C + 10k G: 2000
    # 15k C + 15k G: 1667
    # 15k C + 20k G: 1500
    # 15k C + 25k G: 1250
    # 15k C + 30k G: 1111
    # 15k C + 35k G: 1000

    # 35k C + 0G: 1500
    # 45k C + 0G: 1111
    # 47.5k C + 0G: 1055