# uncond all color baseline
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode clean --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-01-22/cifar10/color1.0_gray0.0/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096/samples_ema/cifar10_uncond_color1.0_gray0.0_epoch_1k_ema_num50000.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_pytorch --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-01-22/cifar10/color1.0_gray0.0/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096/samples_ema/cifar10_uncond_color1.0_gray0.0_epoch_1k_ema_num50000.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-01-22/cifar10/color1.0_gray0.0/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096/samples_ema/cifar10_uncond_color1.0_gray0.0_epoch_1k_ema_num50000.npz


# cond all color baseline
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode clean --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-01-22/cifar10/color1.0_gray0.0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096/samples_ema/cifar10_cond_color1.0_gray0.0_epoch_1k_ema_num50000.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_pytorch --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-01-22/cifar10/color1.0_gray0.0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096/samples_ema/cifar10_cond_color1.0_gray0.0_epoch_1k_ema_num50000.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-01-22/cifar10/color1.0_gray0.0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096/samples_ema/cifar10_cond_color1.0_gray0.0_epoch_1k_ema_num50000.npz

# uncond all gray baseline
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode clean --batch-size 500 --num-gpus 1 --real-mode gray \
--fake ./logs/2023-01-22/cifar10/color0.0_gray1.0/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096/samples_ema/cifar10_uncond_color0.0_gray1.0_epoch_1k_ema_num50000.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_pytorch --batch-size 500 --num-gpus 1 --real-mode gray \
--fake ./logs/2023-01-22/cifar10/color0.0_gray1.0/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096/samples_ema/cifar10_uncond_color0.0_gray1.0_epoch_1k_ema_num50000.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode gray \
--fake ./logs/2023-01-22/cifar10/color0.0_gray1.0/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096/samples_ema/cifar10_uncond_color0.0_gray1.0_epoch_1k_ema_num50000.npz

# cond all gray baseline
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode clean --batch-size 500 --num-gpus 1 --real-mode gray \
--fake ./logs/2023-01-22/cifar10/color0.0_gray1.0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096/samples_ema/cifar10_cond_color0.0_gray1.0_epoch_1k_ema_num50000.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_pytorch --batch-size 500 --num-gpus 1 --real-mode gray \
--fake ./logs/2023-01-22/cifar10/color0.0_gray1.0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096/samples_ema/cifar10_cond_color0.0_gray1.0_epoch_1k_ema_num50000.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode gray \
--fake ./logs/2023-01-22/cifar10/color0.0_gray1.0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096/samples_ema/cifar10_cond_color0.0_gray1.0_epoch_1k_ema_num50000.npz


# 95C-05G
# uncond
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-01-22/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096/samples_ema/cifar10_uncond_color0.95_gray0.05_epoch1000_ema_num50000_color.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode gray \
--fake ./logs/2023-01-22/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096/samples_ema/cifar10_uncond_color0.95_gray0.05_epoch1000_ema_num50000_gray.npz

# cond
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-01-22/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096/samples_ema/cifar10_cond_color0.95_gray0.05_epoch1000_ema_num50000_color.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode gray \
--fake ./logs/2023-01-22/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096/samples_ema/cifar10_cond_color0.95_gray0.05_epoch1000_ema_num50000_gray.npz


# 90C-10G
# uncond
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-01-22/cifar10/color0.9_gray0.1/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096/samples_ema/cifar10_uncond_color0.9_gray0.1_epoch1000_ema_num50000_color.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode gray \
--fake ./logs/2023-01-22/cifar10/color0.9_gray0.1/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096/samples_ema/cifar10_uncond_color0.9_gray0.1_epoch1000_ema_num50000_gray.npz

# cond
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-01-22/cifar10/color0.9_gray0.1/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096/samples_ema/cifar10_cond_color0.9_gray0.1_epoch1000_ema_num50000_color.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode gray \
--fake ./logs/2023-01-22/cifar10/color0.9_gray0.1/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096/samples_ema/cifar10_cond_color0.9_gray0.1_epoch1000_ema_num50000_gray.npz


# 10C-90G
# uncond
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-01-22/cifar10/color0.1_gray0.9/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096/samples_ema/cifar10_uncond_color0.1_gray0.9_epoch1000_ema_num50000_color.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode gray \
--fake ./logs/2023-01-22/cifar10/color0.1_gray0.9/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096/samples_ema/cifar10_uncond_color0.1_gray0.9_epoch1000_ema_num50000_gray.npz

# cond
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-01-22/cifar10/color0.1_gray0.9/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096/samples_ema/cifar10_cond_color0.1_gray0.9_epoch1000_ema_num50000_color.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode gray \
--fake ./logs/2023-01-22/cifar10/color0.1_gray0.9/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096/samples_ema/cifar10_cond_color0.1_gray0.9_epoch1000_ema_num50000_gray.npz


# 05C-95G
# uncond
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-01-22/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096/samples_ema/cifar10_uncond_color0.05_gray0.95_epoch1000_ema_num50000_color.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode gray \
--fake ./logs/2023-01-22/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096/samples_ema/cifar10_uncond_color0.05_gray0.95_epoch1000_ema_num50000_gray.npz

# cond
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-01-22/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096/samples_ema/cifar10_cond_color0.05_gray0.95_epoch1000_ema_num50000_color.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode gray \
--fake ./logs/2023-01-22/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096/samples_ema/cifar10_cond_color0.05_gray0.95_epoch1000_ema_num50000_gray.npz

