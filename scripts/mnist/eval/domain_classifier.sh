#!/bin/sh

#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=4     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_2080:1          # the number of GPUs requested
#SBATCH --mem=24G             # memory 
#SBATCH --output=/n/fs/xl-diffbia/projects/minimal-diffusion/slurm_output/2023-04-08/mnist_flip_domain_classifier_eval_real.txt            # where stdout and stderr will write to
#SBATCH -t 6:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=xl9353@cs.princeton.edu

source ~/.bashrc
conda activate diffusion-bias
cd /n/fs/xl-diffbia/projects/minimal-diffusion

for lr in 0.01 0.005 0.001 0.0005 0.0001; do 
    for wd in 0.001 0.0001 0.00001; do 
        CUDA_VISIBLE_DEVICES=0 python -m domain_classifier.mnist_flip \
        --dataset mnist-subset --num-classes 7 --num-domains 2 --mode test-real \
        --batch-size 2048 --learning-rate $lr --weight-decay $wd \
        --num-gpus 1 --date 2023-04-08 \
        --ckpt-path ./logs/2023-04-08/mnist-subset/domain_classifier/bs256_lr${lr}_decay${wd}/ckpt/model_param_final.pth
    done;
done

CUDA_VISIBLE_DEVICES=0 python -m domain_classifier.mnist_flip \
        --dataset mnist-subset --num-classes 7 --num-domains 2 --mode test-fake \
        --batch-size 2048 --learning-rate 0.001 --weight-decay 0.0001 \
        --num-gpus 1 --date 2023-04-08 \
        --ckpt-path ./logs/2023-04-08/mnist-subset/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth