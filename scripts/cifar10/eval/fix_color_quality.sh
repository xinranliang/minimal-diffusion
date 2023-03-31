# 15k C
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--fake ./logs/2023-03-27/cifar10/color15000_gray0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color15_gray0_ema_num50000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--fake ./logs/2023-03-27/cifar10/color15000_gray5000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color15_gray5_ema_num50000_color.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--fake ./logs/2023-03-27/cifar10/color15000_gray10000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color15_gray10_ema_num50000_color.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--fake ./logs/2023-03-27/cifar10/color15000_gray15000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color15_gray15_ema_num50000_color.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--fake ./logs/2023-03-27/cifar10/color15000_gray20000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color15_gray20_ema_num50000_color.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--fake ./logs/2023-03-27/cifar10/color15000_gray25000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color15_gray25_ema_num50000_color.npz

python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--fake ./logs/2023-03-27/cifar10/color15000_gray30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color15_gray30_ema_num50000_color.npz
