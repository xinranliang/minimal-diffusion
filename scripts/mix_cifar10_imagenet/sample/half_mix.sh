for num in 10000 120000; do 
    # run 1

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
        --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images $num --sampling-color-only --sampling-steps 250 \
        --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 10000 --grayscale 10000 --fix half \
        --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
        --ckpt-name mix_cifar10_imagenet_cond_half10_ema \
        --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/half10000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
        --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images $num --sampling-color-only --sampling-steps 250 \
        --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 20000 --grayscale 20000 --fix half \
        --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
        --ckpt-name mix_cifar10_imagenet_cond_half20_ema \
        --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/half20000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
        --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images $num --sampling-color-only --sampling-steps 250 \
        --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 30000 --grayscale 30000 --fix half \
        --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
        --ckpt-name mix_cifar10_imagenet_cond_half30_ema \
        --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/half30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
        --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images $num --sampling-color-only --sampling-steps 250 \
        --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 45000 --grayscale 45000 --fix half \
        --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
        --ckpt-name mix_cifar10_imagenet_cond_half45_ema \
        --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/half45000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
        --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images $num --sampling-color-only --sampling-steps 250 \
        --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 60000 --grayscale 60000 --fix half \
        --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
        --ckpt-name mix_cifar10_imagenet_cond_half60_ema \
        --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/half60000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
        --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images $num --sampling-color-only --sampling-steps 250 \
        --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 90000 --grayscale 90000 --fix half \
        --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
        --ckpt-name mix_cifar10_imagenet_cond_half90_ema \
        --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/half90000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
        --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images $num --sampling-color-only --sampling-steps 250 \
        --data-dir ./datasets --save-dir ./logs/ --date 2023-04-02 --color 120000 --grayscale 120000 --fix half \
        --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
        --ckpt-name mix_cifar10_imagenet_cond_half120_ema \
        --pretrained-ckpt ./logs/2023-04-02/mix-cifar10-imagenet/half120000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth
    
    # run 2

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
        --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images $num --sampling-color-only --sampling-steps 250 \
        --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 10000 --grayscale 10000 --fix half \
        --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
        --ckpt-name mix_cifar10_imagenet_cond_half10_ema \
        --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/half10000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
        --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images $num --sampling-color-only --sampling-steps 250 \
        --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 20000 --grayscale 20000 --fix half \
        --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
        --ckpt-name mix_cifar10_imagenet_cond_half20_ema \
        --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/half20000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
        --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images $num --sampling-color-only --sampling-steps 250 \
        --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 30000 --grayscale 30000 --fix half \
        --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
        --ckpt-name mix_cifar10_imagenet_cond_half30_ema \
        --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/half30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
        --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images $num --sampling-color-only --sampling-steps 250 \
        --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 45000 --grayscale 45000 --fix half \
        --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
        --ckpt-name mix_cifar10_imagenet_cond_half45_ema \
        --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/half45000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
        --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images $num --sampling-color-only --sampling-steps 250 \
        --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 60000 --grayscale 60000 --fix half \
        --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
        --ckpt-name mix_cifar10_imagenet_cond_half60_ema \
        --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/half60000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
        --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images $num --sampling-color-only --sampling-steps 250 \
        --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 90000 --grayscale 90000 --fix half \
        --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
        --ckpt-name mix_cifar10_imagenet_cond_half90_ema \
        --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/half90000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
        --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images $num --sampling-color-only --sampling-steps 250 \
        --data-dir ./datasets --save-dir ./logs/ --date 2023-04-03 --color 120000 --grayscale 120000 --fix half \
        --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
        --ckpt-name mix_cifar10_imagenet_cond_half120_ema \
        --pretrained-ckpt ./logs/2023-04-03/mix-cifar10-imagenet/half120000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth

done
