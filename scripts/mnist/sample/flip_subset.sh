#!/bin/sh

#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=16     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_2080:4          # the number of GPUs requested
#SBATCH --mem=24G             # memory 
#SBATCH --output=/n/fs/xl-diffbia/projects/minimal-diffusion/slurm_output/2023-04-08/mnist_sample_test.txt            # where stdout and stderr will write to
#SBATCH -t 12:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=xl9353@cs.princeton.edu

source ~/.bashrc
conda activate diffusion-bias
cd /n/fs/xl-diffbia/projects/minimal-diffusion

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 main.py \
    --arch UNetSmall --dataset mnist-subset \
    --data-dir ./datasets --diffusion-steps 1000 --sampling-steps 250 --num-sampled-images 50000 --batch-size 5000 --guidance --sampling-only \
    --class-cond --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-08 --flip-left 0.5 --flip-right 0.5 \
    --ckpt-name mnist_cond_left0.5_right0.5_puncond0.1_epoch100_ema \
    --pretrained-ckpt ./logs/2023-04-08/mnist-subset/left0.5_right0.5/UNetSmall_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_99_ema_0.9995.pth