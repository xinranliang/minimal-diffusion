python metrics-all/compute_metrics.py --dataset cifar10 --resolution 32 --mode legacy_tensorflow --batch-size 500 --num-gpus 1 \
--fake ./logs/2022-12-08/cifar10/color1.0_gray0.0/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096/samples/cifar10_color1_gray0_epoch_1000_num50000.npz
# clean: 12.699; pytorch: 11.807; tf: 11.754