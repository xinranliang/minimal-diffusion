CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8105 main.py \
    --arch UNet --dataset cifar10 --epochs 500 --batch-size 256 --lr 1e-4 --sampling-steps 250 \
    --data-dir datasets/cifar_color --diffusion-steps 1000 \
    --save-dir ./logs/cifar10_color --date 2022-12-01 