# run 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --samplin-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 0 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray0_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/color15000_gray0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_4249_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 7500 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray7.5_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/color15000_gray7500/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_2849_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 15000 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray15_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/color15000_gray15000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_2124_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 22500 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray22.5_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/color15000_gray22500/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1699_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 30000 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray30_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/color15000_gray30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1419_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 60000 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray60_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/color15000_gray60000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_849_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 90000 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray90_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/color15000_gray90000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_609_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 120000 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray120_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/color15000_gray120000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_474_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 150000 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray150_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/color15000_gray150000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_389_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 180000 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray180_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/color15000_gray180000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_329_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 210000 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray210_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/color15000_gray210000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_284_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 240000 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray240_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/color15000_gray240000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_249_ema_0.9995.pth

# run 2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --samplin-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 0 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray0_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/color15000_gray0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_4249_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 7500 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray7.5_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/color15000_gray7500/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_2849_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 15000 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray15_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/color15000_gray15000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_2124_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 22500 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray22.5_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/color15000_gray22500/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1699_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 30000 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray30_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/color15000_gray30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1419_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 60000 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray60_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/color15000_gray60000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_849_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 90000 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray90_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/color15000_gray90000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_609_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 120000 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray120_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/color15000_gray120000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_474_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 150000 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray150_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/color15000_gray150000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_389_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 180000 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray180_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/color15000_gray180000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_329_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 210000 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray210_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/color15000_gray210000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_284_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-color-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 240000 --fix color \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_color15_gray240_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/color15000_gray240000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_249_ema_0.9995.pth
