CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 guidance.py \
    --arch UNet --dataset cifar-imagenet-check --epochs 1000 --batch-size 2500 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --ema_w 0.9995 --ckpt-sample-freq 200 \
    --class-cond --class-cond-dropout 0.1 --num-cifar 0.5 --num-imagenet 0.5 --save-dir ./logs/ --date 2023-07-31 \
    --sampling-only --num-sampled-images 2500 \
    --ckpt-name domaineval_cifar0.5_imagenet0.5_ema \
    --pretrained-ckpt ./logs/2023-07-31/cifar-imagenet-check/cifar0.5_imagenet0.5_numclass20/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 guidance.py \
    --arch UNet --dataset cifar-imagenet-check --epochs 1000 --batch-size 2500 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --ema_w 0.9995 --ckpt-sample-freq 200 \
    --class-cond --class-cond-dropout 0.1 --num-cifar 0.1 --num-imagenet 0.9 --save-dir ./logs/ --date 2023-07-31 \
    --sampling-only --num-sampled-images 2500 \
    --ckpt-name domaineval_cifar0.1_imagenet0.9_ema \
    --pretrained-ckpt ./logs/2023-07-31/cifar-imagenet-check/cifar0.1_imagenet0.9_numclass20/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 guidance.py \
    --arch UNet --dataset cifar-imagenet-check --epochs 1000 --batch-size 2500 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --ema_w 0.9995 --ckpt-sample-freq 200 \
    --class-cond --class-cond-dropout 0.1 --num-cifar 0.9 --num-imagenet 0.1 --save-dir ./logs/ --date 2023-07-31 \
    --sampling-only --num-sampled-images 2500 \
    --ckpt-name domaineval_cifar0.9_imagenet0.1_ema \
    --pretrained-ckpt ./logs/2023-07-31/cifar-imagenet-check/cifar0.9_imagenet0.1_numclass20/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth
