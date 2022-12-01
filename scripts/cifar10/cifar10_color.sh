CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8105 /n/fs/xl-diffbia/projects/minimal-diffusion/main.py \
    --arch UNet --dataset cifar10 --epochs 500 --batch-size 128 --lr 1e-4 --sampling-steps 100 \
    --data-dir /n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar_color \
    --save-dir /n/fs/xl-diffbia/projects/minimal-diffusion/logs/ --date 2022-11-30