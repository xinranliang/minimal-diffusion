CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 main.py \
    --arch UNet --dataset cifar10 --batch-size 512 --num-sampled-images 50000 --sampling-only \
    --data-dir ./datasets --save-dir ./logs/ --date 2022-12-08 \
    --color $1 --grayscale $2 \
    --ckpt-name cifar10_color0.9_gray0.1_epoch_65 \
    --pretrained-ckpt ./logs/2022-12-08/cifar10/UNet_diffusion_1000_sample_250_condition_False_lr_0.0001_bs_256/ckpt/epoch_65.pth 