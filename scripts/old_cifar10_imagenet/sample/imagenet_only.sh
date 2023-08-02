# run 1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-06 --fix none --num-cifar10 0 --num-imagenet 50000 \
    --ckpt-name cifar0_imagenet50_cond_ema \
    --pretrained-ckpt ./logs/2023-04-06/cifar10-imagenet/cifar0_imagenet50000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_999_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-06 --fix none --num-cifar10 0 --num-imagenet 210000 \
    --ckpt-name cifar0_imagenet210_cond_ema \
    --pretrained-ckpt ./logs/2023-04-06/cifar10-imagenet/cifar0_imagenet210000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_299_ema_0.9995.pth

# run 2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-07 --fix none --num-cifar10 0 --num-imagenet 50000 \
    --ckpt-name cifar0_imagenet50_cond_ema \
    --pretrained-ckpt ./logs/2023-04-07/cifar10-imagenet/cifar0_imagenet50000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_999_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-07 --fix none --num-cifar10 0 --num-imagenet 210000 \
    --ckpt-name cifar0_imagenet210_cond_ema \
    --pretrained-ckpt ./logs/2023-04-07/cifar10-imagenet/cifar0_imagenet210000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_299_ema_0.9995.pth
