CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8105 main.py \
    --arch UNet --dataset cifar10 --epochs 1000 --batch-size 512 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 \
    --save-dir ./logs/ --date 2022-12-08 \
    --color $1 --grayscale $2