#!/bin/sh

#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=4     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_3090:1          # the number of GPUs requested
#SBATCH --mem=24G             # memory 
#SBATCH --output=/n/fs/xl-diffbia/projects/minimal-diffusion/slurm_output/2023-07-25/fairface_gender_eval_cond_droprob0.1_subset.txt            # where stdout and stderr will write to
#SBATCH -t 6:00:00           # time requested in hour:minute:second

source ~/.bashrc
conda activate diffusion-bias
cd /n/fs/xl-diffbia/projects/minimal-diffusion

# run 1

CUDA_VISIBLE_DEVICES=0 python guidance.py \
    --arch UNet --dataset fairface --epochs 1000 --batch-size 1000 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --save-dir ./logs/ --date 2023-07-25 --female-ratio 0.1 --male-ratio 0.9 \
    --class-cond --class-cond-dropout 0.1 --ema_w 0.9995 --ckpt-sample-freq 100 \
    --num-sampled-images 50000 --ckpt-name fairface_gendersubset_f0.1_m0.9_ema \
    --pretrained-ckpt ./logs/2023-07-25/fairface/female0.1_male0.9/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_128_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth  \
    --domain-classifier ./logs/2023-07-20/fairface/domain_classifier/bs64_lr0.0001_decay1e-05/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python guidance.py \
    --arch UNet --dataset fairface --epochs 1000 --batch-size 1000 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --save-dir ./logs/ --date 2023-07-25 --female-ratio 0.3 --male-ratio 0.7 \
    --class-cond --class-cond-dropout 0.1 --ema_w 0.9995 --ckpt-sample-freq 100 \
    --num-sampled-images 50000 --ckpt-name fairface_gendersubset_f0.3_m0.7_ema \
    --pretrained-ckpt ./logs/2023-07-25/fairface/female0.3_male0.7/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_128_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-07-20/fairface/domain_classifier/bs64_lr0.0001_decay1e-05/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python guidance.py \
    --arch UNet --dataset fairface --epochs 1000 --batch-size 1000 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --save-dir ./logs/ --date 2023-07-25 --female-ratio 0.5 --male-ratio 0.5 \
    --class-cond --class-cond-dropout 0.1 --ema_w 0.9995 --ckpt-sample-freq 100 \
    --num-sampled-images 50000 --ckpt-name fairface_gendersubset_f0.5_m0.5_ema \
    --pretrained-ckpt ./logs/2023-07-25/fairface/female0.5_male0.5/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_128_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-07-20/fairface/domain_classifier/bs64_lr0.0001_decay1e-05/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python guidance.py \
    --arch UNet --dataset fairface --epochs 1000 --batch-size 1000 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --save-dir ./logs/ --date 2023-07-25 --female-ratio 0.7 --male-ratio 0.3 \
    --class-cond --class-cond-dropout 0.1 --ema_w 0.9995 --ckpt-sample-freq 100 \
    --num-sampled-images 50000 --ckpt-name fairface_gendersubset_f0.7_m0.3_ema \
    --pretrained-ckpt ./logs/2023-07-25/fairface/female0.7_male0.3/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_128_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-07-20/fairface/domain_classifier/bs64_lr0.0001_decay1e-05/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python guidance.py \
    --arch UNet --dataset fairface --epochs 1000 --batch-size 1000 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --save-dir ./logs/ --date 2023-07-25 --female-ratio 0.9 --male-ratio 0.1 \
    --class-cond --class-cond-dropout 0.1 --ema_w 0.9995 --ckpt-sample-freq 100 \
    --num-sampled-images 50000 --ckpt-name fairface_gendersubset_f0.9_m0.1_ema \
    --pretrained-ckpt ./logs/2023-07-25/fairface/female0.9_male0.1/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_128_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-07-20/fairface/domain_classifier/bs64_lr0.0001_decay1e-05/ckpt/model_param_final.pth


# run 2

CUDA_VISIBLE_DEVICES=0 python guidance.py \
    --arch UNet --dataset fairface --epochs 1000 --batch-size 1000 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --save-dir ./logs/ --date 2023-07-26 --female-ratio 0.1 --male-ratio 0.9 \
    --class-cond --class-cond-dropout 0.1 --ema_w 0.9995 --ckpt-sample-freq 100 \
    --num-sampled-images 50000 --ckpt-name fairface_gendersubset_f0.1_m0.9_ema \
    --pretrained-ckpt ./logs/2023-07-26/fairface/female0.1_male0.9/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_128_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth  \
    --domain-classifier ./logs/2023-07-20/fairface/domain_classifier/bs64_lr0.0001_decay1e-05/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python guidance.py \
    --arch UNet --dataset fairface --epochs 1000 --batch-size 1000 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --save-dir ./logs/ --date 2023-07-26 --female-ratio 0.3 --male-ratio 0.7 \
    --class-cond --class-cond-dropout 0.1 --ema_w 0.9995 --ckpt-sample-freq 100 \
    --num-sampled-images 50000 --ckpt-name fairface_gendersubset_f0.3_m0.7_ema \
    --pretrained-ckpt ./logs/2023-07-26/fairface/female0.3_male0.7/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_128_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-07-20/fairface/domain_classifier/bs64_lr0.0001_decay1e-05/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python guidance.py \
    --arch UNet --dataset fairface --epochs 1000 --batch-size 1000 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --save-dir ./logs/ --date 2023-07-26 --female-ratio 0.5 --male-ratio 0.5 \
    --class-cond --class-cond-dropout 0.1 --ema_w 0.9995 --ckpt-sample-freq 100 \
    --num-sampled-images 50000 --ckpt-name fairface_gendersubset_f0.5_m0.5_ema \
    --pretrained-ckpt ./logs/2023-07-26/fairface/female0.5_male0.5/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_128_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-07-20/fairface/domain_classifier/bs64_lr0.0001_decay1e-05/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python guidance.py \
    --arch UNet --dataset fairface --epochs 1000 --batch-size 1000 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --save-dir ./logs/ --date 2023-07-26 --female-ratio 0.7 --male-ratio 0.3 \
    --class-cond --class-cond-dropout 0.1 --ema_w 0.9995 --ckpt-sample-freq 100 \
    --num-sampled-images 50000 --ckpt-name fairface_gendersubset_f0.7_m0.3_ema \
    --pretrained-ckpt ./logs/2023-07-26/fairface/female0.7_male0.3/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_128_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-07-20/fairface/domain_classifier/bs64_lr0.0001_decay1e-05/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python guidance.py \
    --arch UNet --dataset fairface --epochs 1000 --batch-size 1000 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --save-dir ./logs/ --date 2023-07-26 --female-ratio 0.9 --male-ratio 0.1 \
    --class-cond --class-cond-dropout 0.1 --ema_w 0.9995 --ckpt-sample-freq 100 \
    --num-sampled-images 50000 --ckpt-name fairface_gendersubset_f0.9_m0.1_ema \
    --pretrained-ckpt ./logs/2023-07-26/fairface/female0.9_male0.1/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_128_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-07-20/fairface/domain_classifier/bs64_lr0.0001_decay1e-05/ckpt/model_param_final.pth
