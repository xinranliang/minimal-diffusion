# 2500C + 0G

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-02-07/cifar10/color2500_gray0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color25_gray0_ema_num50000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-02-07/cifar10/color2500_gray0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color25_gray0_ema_num50000_guidance0.5.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-02-07/cifar10/color2500_gray0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color25_gray0_ema_num50000_guidance1.0.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-02-07/cifar10/color2500_gray0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color25_gray0_ema_num50000_guidance2.0.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-02-07/cifar10/color2500_gray0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color25_gray0_ema_num50000_guidance4.0.npz
