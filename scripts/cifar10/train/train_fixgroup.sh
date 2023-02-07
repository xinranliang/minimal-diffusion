# number of gradient steps ~ 200k
# compute number of training epochs:
    # 2500C + 0G: 20001
    # 2500C + 500G: 16001
    # 2500C + 1000G: 14001
    # 2500C + 1500G: 13001
    # 2500C + 2000G: 11001
    # 2500C + 2500G: 10001
    # 2500C + 5000G: 7001
    # 2500C + 10000G: 4001
    # 2500C + 20000G: 2201
    # 2500C + 40000G: 1201

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port $4 main.py \
    --arch UNet --dataset cifar10 --epochs $2 --batch-size 256 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --ema_w 0.9995 --ckpt-sample-freq $3 \
    --save-dir ./logs/ --date 2023-02-07 \
    --color 2500 --grayscale $1 --fix subgroup

# --class-cond --class-cond-dropout 0.1 \