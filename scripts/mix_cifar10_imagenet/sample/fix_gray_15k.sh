# run 1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 0 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color0_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/gray15000_color0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_4249_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 7500 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color7.5_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/gray15000_color7500/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_2849_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 15000 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color15_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/gray15000_color15000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_2124_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 22500 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color22.5_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/gray15000_color22500/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1699_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 30000 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color30_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/gray15000_color30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1419_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 60000 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color60_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/gray15000_color60000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_849_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 90000 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color90_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/gray15000_color90000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_609_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 120000 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color120_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/gray15000_color120000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_474_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 150000 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color150_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/gray15000_color150000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_389_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 180000 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color180_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/gray15000_color180000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_329_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 210000 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color210_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/gray15000_color210000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_284_ema_0.9995.pth
# not done
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 15000 --grayscale 240000 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color240_ema \
    --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/gray15000_color240000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_249_ema_0.9995.pth

# run 2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 0 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color0_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/gray15000_color0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_4249_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 7500 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color7.5_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/gray15000_color7500/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_2849_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 15000 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color15_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/gray15000_color15000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_2124_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 22500 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color22.5_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/gray15000_color22500/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1699_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 30000 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color30_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/gray15000_color30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1419_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 60000 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color60_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/gray15000_color60000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_849_ema_0.9995.pth
# not done
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 90000 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color90_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/gray15000_color90000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_609_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 120000 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color120_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/gray15000_color120000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_474_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 150000 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color150_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/gray15000_color150000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_389_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 180000 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color180_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/gray15000_color180000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_329_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 210000 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color210_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/gray15000_color210000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_284_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 15000 --sampling-gray-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 15000 --grayscale 240000 --fix gray \
    --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --ckpt-name mix_cifar10_imagenet_cond_gray15_color240_ema \
    --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/gray15000_color240000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_249_ema_0.9995.pth
