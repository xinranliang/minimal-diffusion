#!/bin/sh

#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=4     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_3090:1          # the number of GPUs requested
#SBATCH --mem=24G             # memory 
#SBATCH --output=/n/fs/xl-diffbia/projects/minimal-diffusion/slurm_output/2023-03-27/cifar10_eval_fixcolor_cond_droprob0.1.txt            # where stdout and stderr will write to
#SBATCH -t 12:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=xl9353@cs.princeton.edu

source ~/.bashrc
conda activate diffusion-bias
cd /n/fs/xl-diffbia/projects/minimal-diffusion


# 15k C
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-03-27 --num-samples 15000 \
--fake ./logs/2023-03-27/cifar10/color15000_gray0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color15_gray0_ema_num50000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-03-27 --num-samples 15000 \
--fake ./logs/2023-03-27/cifar10/color15000_gray5000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color15_gray5_ema_num50000_color.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-03-27 --num-samples 15000 \
--fake ./logs/2023-03-27/cifar10/color15000_gray10000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color15_gray10_ema_num50000_color.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-03-27 --num-samples 15000 \
--fake ./logs/2023-03-27/cifar10/color15000_gray15000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color15_gray15_ema_num50000_color.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-03-27 --num-samples 15000 \
--fake ./logs/2023-03-27/cifar10/color15000_gray20000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color15_gray20_ema_num50000_color.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-03-27 --num-samples 15000 \
--fake ./logs/2023-03-27/cifar10/color15000_gray25000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color15_gray25_ema_num50000_color.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-03-27 --num-samples 15000 \
--fake ./logs/2023-03-27/cifar10/color15000_gray30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color15_gray30_ema_num50000_color.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-03-27 --num-samples 15000 \
--fake ./logs/2023-03-27/cifar10/color0.3_gray0.7/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_10000_dropprob_0.1/samples_ema/cifar10_cond_color0.3_gray0.7_puncond0.1_epoch1k_ema_num50000_color.npz