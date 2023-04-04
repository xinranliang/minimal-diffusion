# run 1

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNet --dataset cifar10 --batch-size 10000 --num-sampled-images 50000 --guidance --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-02-06 --color 0.95 --grayscale 0.05 --fix total \
    --class-cond --class-cond-dropout 0.1 --classifier-free-w 0.0 --diffusion-steps 1000 \
    --ckpt-name cifar10_cond_color0.95_gray0.05_puncond0.1_epoch1k_ema \
    --pretrained-ckpt ./logs/2023-02-06/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_1000_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1000_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNet --dataset cifar10 --batch-size 10000 --num-sampled-images 50000 --guidance --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-02-06 --color 0.9 --grayscale 0.1 --fix total \
    --class-cond --class-cond-dropout 0.1 --classifier-free-w 0.0 --diffusion-steps 1000 \
    --ckpt-name cifar10_cond_color0.9_gray0.1_puncond0.1_epoch1k_ema \
    --pretrained-ckpt ./logs/2023-02-06/cifar10/color0.9_gray0.1/UNet_diffusionstep_1000_samplestep_1000_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1000_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNet --dataset cifar10 --batch-size 10000 --num-sampled-images 50000 --guidance --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-02-06 --color 0.7 --grayscale 0.3 --fix total \
    --class-cond --class-cond-dropout 0.1 --classifier-free-w 0.0 --diffusion-steps 1000 \
    --ckpt-name cifar10_cond_color0.7_gray0.3_puncond0.1_epoch1k_ema \
    --pretrained-ckpt ./logs/2023-02-06/cifar10/color0.7_gray0.3/UNet_diffusionstep_1000_samplestep_1000_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1000_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNet --dataset cifar10 --batch-size 10000 --num-sampled-images 50000 --guidance --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-02-06 --color 0.5 --grayscale 0.5 --fix total \
    --class-cond --class-cond-dropout 0.1 --classifier-free-w 0.0 --diffusion-steps 1000 \
    --ckpt-name cifar10_cond_color0.5_gray0.5_puncond0.1_epoch1k_ema \
    --pretrained-ckpt ./logs/2023-02-06/cifar10/color0.5_gray0.5/UNet_diffusionstep_1000_samplestep_1000_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1000_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNet --dataset cifar10 --batch-size 10000 --num-sampled-images 50000 --guidance --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-02-06 --color 0.3 --grayscale 0.7 --fix total \
    --class-cond --class-cond-dropout 0.1 --classifier-free-w 0.0 --diffusion-steps 1000 \
    --ckpt-name cifar10_cond_color0.3_gray0.7_puncond0.1_epoch1k_ema \
    --pretrained-ckpt ./logs/2023-02-06/cifar10/color0.3_gray0.7/UNet_diffusionstep_1000_samplestep_1000_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1000_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNet --dataset cifar10 --batch-size 10000 --num-sampled-images 50000 --guidance --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-02-06 --color 0.1 --grayscale 0.9 --fix total \
    --class-cond --class-cond-dropout 0.1 --classifier-free-w 0.0 --diffusion-steps 1000 \
    --ckpt-name cifar10_cond_color0.1_gray0.9_puncond0.1_epoch1k_ema \
    --pretrained-ckpt ./logs/2023-02-06/cifar10/color0.1_gray0.9/UNet_diffusionstep_1000_samplestep_1000_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1000_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNet --dataset cifar10 --batch-size 10000 --num-sampled-images 50000 --guidance --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-02-06 --color 0.05 --grayscale 0.95 --fix total \
    --class-cond --class-cond-dropout 0.1 --classifier-free-w 0.0 --diffusion-steps 1000 \
    --ckpt-name cifar10_cond_color0.05_gray0.95_puncond0.1_epoch1k_ema \
    --pretrained-ckpt ./logs/2023-02-06/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_1000_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1000_ema_0.9995.pth


# run 2

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNet --dataset cifar10 --batch-size 10000 --num-sampled-images 50000 --guidance --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-03-27 --color 0.95 --grayscale 0.05 --fix total \
    --class-cond --class-cond-dropout 0.1 --classifier-free-w 0.0 --diffusion-steps 1000 \
    --ckpt-name cifar10_cond_color0.95_gray0.05_puncond0.1_epoch1k_ema \
    --pretrained-ckpt ./logs/2023-03-27/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1000_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNet --dataset cifar10 --batch-size 10000 --num-sampled-images 50000 --guidance --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-03-27 --color 0.9 --grayscale 0.1 --fix total \
    --class-cond --class-cond-dropout 0.1 --classifier-free-w 0.0 --diffusion-steps 1000 \
    --ckpt-name cifar10_cond_color0.9_gray0.1_puncond0.1_epoch1k_ema \
    --pretrained-ckpt ./logs/2023-03-27/cifar10/color0.9_gray0.1/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1000_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNet --dataset cifar10 --batch-size 10000 --num-sampled-images 50000 --guidance --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-03-27 --color 0.7 --grayscale 0.3 --fix total \
    --class-cond --class-cond-dropout 0.1 --classifier-free-w 0.0 --diffusion-steps 1000 \
    --ckpt-name cifar10_cond_color0.7_gray0.3_puncond0.1_epoch1k_ema \
    --pretrained-ckpt ./logs/2023-03-27/cifar10/color0.7_gray0.3/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1000_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNet --dataset cifar10 --batch-size 10000 --num-sampled-images 50000 --guidance --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-03-27 --color 0.5 --grayscale 0.5 --fix total \
    --class-cond --class-cond-dropout 0.1 --classifier-free-w 0.0 --diffusion-steps 1000 \
    --ckpt-name cifar10_cond_color0.5_gray0.5_puncond0.1_epoch1k_ema \
    --pretrained-ckpt ./logs/2023-03-27/cifar10/color0.5_gray0.5/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1000_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNet --dataset cifar10 --batch-size 10000 --num-sampled-images 50000 --guidance --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-03-27 --color 0.3 --grayscale 0.7 --fix total \
    --class-cond --class-cond-dropout 0.1 --classifier-free-w 0.0 --diffusion-steps 1000 \
    --ckpt-name cifar10_cond_color0.3_gray0.7_puncond0.1_epoch1k_ema \
    --pretrained-ckpt ./logs/2023-03-27/cifar10/color0.3_gray0.7/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1000_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNet --dataset cifar10 --batch-size 10000 --num-sampled-images 50000 --guidance --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-03-27 --color 0.1 --grayscale 0.9 --fix total \
    --class-cond --class-cond-dropout 0.1 --classifier-free-w 0.0 --diffusion-steps 1000 \
    --ckpt-name cifar10_cond_color0.1_gray0.9_puncond0.1_epoch1k_ema \
    --pretrained-ckpt ./logs/2023-03-27/cifar10/color0.1_gray0.9/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1000_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNet --dataset cifar10 --batch-size 10000 --num-sampled-images 50000 --guidance --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-03-27 --color 0.05 --grayscale 0.95 --fix total \
    --class-cond --class-cond-dropout 0.1 --classifier-free-w 0.0 --diffusion-steps 1000 \
    --ckpt-name cifar10_cond_color0.05_gray0.95_puncond0.1_epoch1k_ema \
    --pretrained-ckpt ./logs/2023-03-27/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1000_ema_0.9995.pth
