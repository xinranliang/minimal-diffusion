# uncond all color baseline
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode clean --batch-size 500 --num-gpus 1 \
--fake ./logs/2023-01-22/cifar10/color1.0_gray0.0/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096/samples/cifar10_uncond_color1.0_gray0.0_epoch_1k_num50000.npz
# clean-fid: 8.619
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_pytorch --batch-size 500 --num-gpus 1 \
--fake ./logs/2023-01-22/cifar10/color1.0_gray0.0/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096/samples/cifar10_uncond_color1.0_gray0.0_epoch_1k_num50000.npz
# pytorch-fid: 8.132
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 \
--fake ./logs/2023-01-22/cifar10/color1.0_gray0.0/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096/samples/cifar10_uncond_color1.0_gray0.0_epoch_1k_num50000.npz
# tf-fid: 8.115

# cond all color baseline
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode clean --batch-size 500 --num-gpus 1 \
--fake ./logs/2023-01-22/cifar10/color1.0_gray0.0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096/samples/cifar10_cond_color1.0_gray0.0_epoch_1k_num50000.npz
# clean-fid: 11.840
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_pytorch --batch-size 500 --num-gpus 1 \
--fake ./logs/2023-01-22/cifar10/color1.0_gray0.0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096/samples/cifar10_cond_color1.0_gray0.0_epoch_1k_num50000.npz
# pytorch-fid: 10.731
python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 \
--fake ./logs/2023-01-22/cifar10/color1.0_gray0.0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096/samples/cifar10_cond_color1.0_gray0.0_epoch_1k_num50000.npz
# tf-fid: 10.662