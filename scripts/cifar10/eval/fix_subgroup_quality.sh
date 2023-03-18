# NO guidance

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-02-21/cifar10/color25000_gray0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color25_gray0_ema_num50000_color.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-02-21/cifar10/color25000_gray5000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color25_gray5_ema_num50000_color.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-02-21/cifar10/color25000_gray10000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color25_gray10_ema_num50000_color.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-02-21/cifar10/color25000_gray15000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color25_gray15_ema_num50000_color.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 --real-mode color \
--fake ./logs/2023-02-21/cifar10/color25000_gray20000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color25_gray20_ema_num50000_color.npz
