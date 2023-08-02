#!/bin/sh

#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=4     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_2080:1          # the number of GPUs requested
#SBATCH --mem=24G             # memory 
#SBATCH --output=/n/fs/xl-diffbia/projects/minimal-diffusion/slurm_output/2023-07-20/celeba_gender_classifier_eval.txt            # where stdout and stderr will write to
#SBATCH -t 24:00:00           # time requested in hour:minute:second

source ~/.bashrc
conda activate diffusion-bias
cd /n/fs/xl-diffbia/projects/minimal-diffusion

for lr in 0.01 0.005 0.001 0.0005 0.0001; do 
    for wd in 0.001 0.0001 0.00001; do 
        CUDA_VISIBLE_DEVICES=0 python -m domain_classifier.celeba \
            --arch resnet50 --pretrained --dataset celeba --num-domains 2 --mode test \
            --num-epochs 20 --batch-size 64 --learning-rate $lr --weight-decay $wd \
            --num-gpus 1 --date 2023-07-20
    done;
done;