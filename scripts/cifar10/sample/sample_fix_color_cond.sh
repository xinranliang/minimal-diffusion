
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 4096 --num-sampled-images 50000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-02-07 --color 25000 --grayscale 0 --fix subgroup \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name cifar10_cond_color25_gray0_ema \
    --pretrained-ckpt ./logs/2023-02-21/cifar10/color25000_gray0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_2000_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 4096 --num-sampled-images 50000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-02-07 --color 25000 --grayscale 5000 --fix subgroup \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name cifar10_cond_color25_gray5_ema \
    --pretrained-ckpt ./logs/2023-02-21/cifar10/color25000_gray5000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1800_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 4096 --num-sampled-images 50000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-02-07 --color 25000 --grayscale 10000 --fix subgroup \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name cifar10_cond_color25_gray10_ema \
    --pretrained-ckpt ./logs/2023-02-21/cifar10/color25000_gray10000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1600_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 4096 --num-sampled-images 50000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-02-07 --color 25000 --grayscale 15000 --fix subgroup \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name cifar10_cond_color25_gray15_ema \
    --pretrained-ckpt ./logs/2023-02-21/cifar10/color25000_gray15000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1400_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 4096 --num-sampled-images 50000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-02-07 --color 25000 --grayscale 20000 --fix subgroup \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name cifar10_cond_color25_gray20_ema \
    --pretrained-ckpt ./logs/2023-02-21/cifar10/color25000_gray20000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1200_ema_0.9995.pth
