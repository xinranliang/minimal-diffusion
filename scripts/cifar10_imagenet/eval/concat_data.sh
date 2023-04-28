#!/bin/sh

#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=4     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_3090:1          # the number of GPUs requested
#SBATCH --mem=24G             # memory 
#SBATCH --output=/n/fs/xl-diffbia/projects/minimal-diffusion/slurm_output/2023-04-06/compute_metrics_concat_data.txt            # where stdout and stderr will write to
#SBATCH -t 12:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=xl9353@cs.princeton.edu

source ~/.bashrc
conda activate diffusion-bias
cd /n/fs/xl-diffbia/projects/minimal-diffusion

# run 1

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color --num-samples 50000 \
--fake ./logs/2023-04-06/cifar10-imagenet/cifar50000_imagenet25000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/cifar50_imagenet25_cond_ema_num50000_cifar.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color --num-samples 50000 \
--fake ./logs/2023-04-06/cifar10-imagenet/cifar50000_imagenet50000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/cifar50_imagenet50_cond_ema_num50000_cifar.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color --num-samples 50000 \
--fake ./logs/2023-04-06/cifar10-imagenet/cifar50000_imagenet75000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/cifar50_imagenet75_cond_ema_num50000_cifar.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color --num-samples 50000 \
--fake ./logs/2023-04-06/cifar10-imagenet/cifar50000_imagenet100000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/cifar50_imagenet100_cond_ema_num50000_cifar.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color --num-samples 50000 \
--fake ./logs/2023-04-06/cifar10-imagenet/cifar50000_imagenet125000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/cifar50_imagenet125_cond_ema_num50000_cifar.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color --num-samples 50000 \
--fake ./logs/2023-04-06/cifar10-imagenet/cifar50000_imagenet150000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/cifar50_imagenet150_cond_ema_num50000_cifar.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color --num-samples 50000 \
--fake ./logs/2023-04-06/cifar10-imagenet/cifar50000_imagenet175000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/cifar50_imagenet175_cond_ema_num50000_cifar.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color --num-samples 50000 \
--fake ./logs/2023-04-06/cifar10-imagenet/cifar50000_imagenet200000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/cifar50_imagenet200_cond_ema_num50000_cifar.npz

# run 2

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color --num-samples 50000 \
--fake ./logs/2023-04-07/cifar10-imagenet/cifar50000_imagenet25000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/cifar50_imagenet25_cond_ema_num50000_cifar.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color --num-samples 50000 \
--fake ./logs/2023-04-07/cifar10-imagenet/cifar50000_imagenet50000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/cifar50_imagenet50_cond_ema_num50000_cifar.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color --num-samples 50000 \
--fake ./logs/2023-04-07/cifar10-imagenet/cifar50000_imagenet75000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/cifar50_imagenet75_cond_ema_num50000_cifar.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color --num-samples 50000 \
--fake ./logs/2023-04-07/cifar10-imagenet/cifar50000_imagenet100000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/cifar50_imagenet100_cond_ema_num50000_cifar.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color --num-samples 50000 \
--fake ./logs/2023-04-07/cifar10-imagenet/cifar50000_imagenet125000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/cifar50_imagenet125_cond_ema_num50000_cifar.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color --num-samples 50000 \
--fake ./logs/2023-04-07/cifar10-imagenet/cifar50000_imagenet150000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/cifar50_imagenet150_cond_ema_num50000_cifar.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color --num-samples 50000 \
--fake ./logs/2023-04-07/cifar10-imagenet/cifar50000_imagenet175000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/cifar50_imagenet175_cond_ema_num50000_cifar.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color --num-samples 50000 \
--fake ./logs/2023-04-07/cifar10-imagenet/cifar50000_imagenet200000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/cifar50_imagenet200_cond_ema_num50000_cifar.npz
