CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 512 --num-sampled-images 50000 --sampling-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2022-12-08 \
    --color 0.01 --grayscale 0.99 \
    --ckpt-name cifar10_color0.01_gray0.99_epoch_950 \
    --pretrained-ckpt ./logs/2022-12-08/cifar10/color0.01_gray0.99/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/ckpt/epoch_950.pth 

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 512 --num-sampled-images 50000 --sampling-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2022-12-08 \
    --color 0.05 --grayscale 0.95 \
    --ckpt-name cifar10_color0.05_gray0.95_epoch_950 \
    --pretrained-ckpt ./logs/2022-12-08/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/ckpt/epoch_950.pth 

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 512 --num-sampled-images 50000 --sampling-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2022-12-08 \
    --color 0.95 --grayscale 0.05 \
    --ckpt-name cifar10_color0.95_gray0.05_epoch_950 \
    --pretrained-ckpt ./logs/2022-12-08/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/ckpt/epoch_950.pth 

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 512 --num-sampled-images 50000 --sampling-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2022-12-08 \
    --color 0.99 --grayscale 0.01 \
    --ckpt-name cifar10_color0.99_gray0.01_epoch_950 \
    --pretrained-ckpt ./logs/2022-12-08/cifar10/color0.99_gray0.01/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/ckpt/epoch_950.pth 

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 512 --num-sampled-images 50000 --sampling-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2022-12-08 \
    --color 1.0 --grayscale 0.0 \
    --ckpt-name cifar10_color1_gray0_epoch_950 \
    --pretrained-ckpt ./logs/2022-12-08/cifar10/color1.0_gray0.0/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/ckpt/epoch_950.pth 