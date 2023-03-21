# color

# 95C-05G
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--fake ./logs/2023-02-06/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color0.95_gray0.05_puncond0.1_epoch1k_ema_num50000_color.npz
# 90C-10G
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--fake ./logs/2023-02-06/cifar10/color0.9_gray0.1/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color0.9_gray0.1_puncond0.1_epoch1k_ema_num50000_color.npz
# 70C-30G
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--fake ./logs/2023-02-06/cifar10/color0.7_gray0.3/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color0.7_gray0.3_puncond0.1_epoch1k_ema_num50000_color.npz
# 50C-50G
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--fake ./logs/2023-02-06/cifar10/color0.5_gray0.5/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color0.5_gray0.5_puncond0.1_epoch1k_ema_num50000_color.npz
# 30C-70G
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--fake ./logs/2023-02-06/cifar10/color0.3_gray0.7/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color0.3_gray0.7_puncond0.1_epoch1k_ema_num50000_color.npz
# 10C-90G
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--fake ./logs/2023-02-06/cifar10/color0.1_gray0.9/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color0.1_gray0.9_puncond0.1_epoch1k_ema_num50000_color.npz
# 05C-95G
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--fake ./logs/2023-02-06/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color0.05_gray0.95_puncond0.1_epoch1k_ema_num50000_color.npz

# gray
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--fake ./logs/2023-02-06/cifar10/color0.0_gray1.0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color0.0_gray1.0_puncond0.1_epoch1k_ema_num50000_guidance0.0.npz
# 05C-95G
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--fake ./logs/2023-02-06/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color0.05_gray0.95_puncond0.1_epoch1k_ema_num50000_gray.npz
# 10C-90G
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--fake ./logs/2023-02-06/cifar10/color0.1_gray0.9/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color0.1_gray0.9_puncond0.1_epoch1k_ema_num50000_gray.npz
# 30C-70G
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--fake ./logs/2023-02-06/cifar10/color0.3_gray0.7/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color0.3_gray0.7_puncond0.1_epoch1k_ema_num50000_gray.npz
# 50C-50G
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--fake ./logs/2023-02-06/cifar10/color0.5_gray0.5/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color0.5_gray0.5_puncond0.1_epoch1k_ema_num50000_gray.npz
# 70C-30G
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--fake ./logs/2023-02-06/cifar10/color0.7_gray0.3/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color0.7_gray0.3_puncond0.1_epoch1k_ema_num50000_gray.npz
# 90C-10G
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--fake ./logs/2023-02-06/cifar10/color0.9_gray0.1/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color0.9_gray0.1_puncond0.1_epoch1k_ema_num50000_gray.npz
# 95C-05G
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--fake ./logs/2023-02-06/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1/samples_ema/cifar10_cond_color0.95_gray0.05_puncond0.1_epoch1k_ema_num50000_gray.npz
