#!/bin/sh

#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=4     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_3090:1          # the number of GPUs requested
#SBATCH --mem=24G             # memory 
#SBATCH --output=/n/fs/xl-diffbia/projects/minimal-diffusion/slurm_output/2023-07-31/eval_cifar_imagenet_cond_droprob0.1.txt            # where stdout and stderr will write to
#SBATCH -t 6:00:00           # time requested in hour:minute:second

source ~/.bashrc
conda activate diffusion-bias
cd /n/fs/xl-diffbia/projects/minimal-diffusion

# run 1

CUDA_VISIBLE_DEVICES=0 python guidance.py \
    --arch UNet --dataset cifar-imagenet --epochs 1000 --batch-size 2500 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --ema_w 0.9995 --ckpt-sample-freq 200 \
    --class-cond --class-cond-dropout 0.1 --num-cifar 0.1 --num-imagenet 0.9 --save-dir ./logs/ --date 2023-07-31 \
    --num-sampled-images 50000 --ckpt-name cifar0.1_imagenet0.9_ema \
    --pretrained-ckpt ./logs/2023-07-31/cifar-imagenet/cifar0.1_imagenet0.9/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python guidance.py \
    --arch UNet --dataset cifar-imagenet --epochs 1000 --batch-size 2500 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --ema_w 0.9995 --ckpt-sample-freq 200 \
    --class-cond --class-cond-dropout 0.1 --num-cifar 0.3 --num-imagenet 0.7 --save-dir ./logs/ --date 2023-07-31 \
    --num-sampled-images 50000 --ckpt-name cifar0.3_imagenet0.7_ema \
    --pretrained-ckpt ./logs/2023-07-31/cifar-imagenet/cifar0.3_imagenet0.7/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python guidance.py \
    --arch UNet --dataset cifar-imagenet --epochs 1000 --batch-size 2500 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --ema_w 0.9995 --ckpt-sample-freq 200 \
    --class-cond --class-cond-dropout 0.1 --num-cifar 0.5 --num-imagenet 0.5 --save-dir ./logs/ --date 2023-07-31 \
    --num-sampled-images 50000 --ckpt-name cifar0.5_imagenet0.5_ema \
    --pretrained-ckpt ./logs/2023-07-31/cifar-imagenet/cifar0.5_imagenet0.5/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python guidance.py \
    --arch UNet --dataset cifar-imagenet --epochs 1000 --batch-size 2500 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --ema_w 0.9995 --ckpt-sample-freq 200 \
    --class-cond --class-cond-dropout 0.1 --num-cifar 0.7 --num-imagenet 0.3 --save-dir ./logs/ --date 2023-07-31 \
    --num-sampled-images 50000 --ckpt-name cifar0.7_imagenet0.3_ema \
    --pretrained-ckpt ./logs/2023-07-31/cifar-imagenet/cifar0.7_imagenet0.3/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python guidance.py \
    --arch UNet --dataset cifar-imagenet --epochs 1000 --batch-size 2500 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --ema_w 0.9995 --ckpt-sample-freq 200 \
    --class-cond --class-cond-dropout 0.1 --num-cifar 0.9 --num-imagenet 0.1 --save-dir ./logs/ --date 2023-07-31 \
    --num-sampled-images 50000 --ckpt-name cifar0.9_imagenet0.1_ema \
    --pretrained-ckpt ./logs/2023-07-31/cifar-imagenet/cifar0.9_imagenet0.1/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth


# run 2

CUDA_VISIBLE_DEVICES=0 python guidance.py \
    --arch UNet --dataset cifar-imagenet --epochs 1000 --batch-size 2500 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --ema_w 0.9995 --ckpt-sample-freq 200 \
    --class-cond --class-cond-dropout 0.1 --num-cifar 0.1 --num-imagenet 0.9 --save-dir ./logs/ --date 2023-08-01 \
    --num-sampled-images 50000 --ckpt-name cifar0.1_imagenet0.9_ema \
    --pretrained-ckpt ./logs/2023-08-01/cifar-imagenet/cifar0.1_imagenet0.9/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python guidance.py \
    --arch UNet --dataset cifar-imagenet --epochs 1000 --batch-size 2500 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --ema_w 0.9995 --ckpt-sample-freq 200 \
    --class-cond --class-cond-dropout 0.1 --num-cifar 0.3 --num-imagenet 0.7 --save-dir ./logs/ --date 2023-08-01 \
    --num-sampled-images 50000 --ckpt-name cifar0.3_imagenet0.7_ema \
    --pretrained-ckpt ./logs/2023-08-01/cifar-imagenet/cifar0.3_imagenet0.7/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python guidance.py \
    --arch UNet --dataset cifar-imagenet --epochs 1000 --batch-size 2500 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --ema_w 0.9995 --ckpt-sample-freq 200 \
    --class-cond --class-cond-dropout 0.1 --num-cifar 0.5 --num-imagenet 0.5 --save-dir ./logs/ --date 2023-08-01 \
    --num-sampled-images 50000 --ckpt-name cifar0.5_imagenet0.5_ema \
    --pretrained-ckpt ./logs/2023-08-01/cifar-imagenet/cifar0.5_imagenet0.5/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python guidance.py \
    --arch UNet --dataset cifar-imagenet --epochs 1000 --batch-size 2500 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --ema_w 0.9995 --ckpt-sample-freq 200 \
    --class-cond --class-cond-dropout 0.1 --num-cifar 0.7 --num-imagenet 0.3 --save-dir ./logs/ --date 2023-08-01 \
    --num-sampled-images 50000 --ckpt-name cifar0.7_imagenet0.3_ema \
    --pretrained-ckpt ./logs/2023-08-01/cifar-imagenet/cifar0.7_imagenet0.3/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python guidance.py \
    --arch UNet --dataset cifar-imagenet --epochs 1000 --batch-size 2500 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --ema_w 0.9995 --ckpt-sample-freq 200 \
    --class-cond --class-cond-dropout 0.1 --num-cifar 0.9 --num-imagenet 0.1 --save-dir ./logs/ --date 2023-08-01 \
    --num-sampled-images 50000 --ckpt-name cifar0.9_imagenet0.1_ema \
    --pretrained-ckpt ./logs/2023-08-01/cifar-imagenet/cifar0.9_imagenet0.1/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth
