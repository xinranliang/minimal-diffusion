# run 1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-06 --fix none --num-cifar10 50000 --num-imagenet 25000 \
    --ckpt-name cifar50_imagenet25_cond_ema \
    --pretrained-ckpt ./logs/2023-04-06/cifar10-imagenet/cifar50000_imagenet25000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_849_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-06 --fix none --num-cifar10 50000 --num-imagenet 50000 \
    --ckpt-name cifar50_imagenet50_cond_ema \
    --pretrained-ckpt ./logs/2023-04-06/cifar10-imagenet/cifar50000_imagenet50000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_624_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-06 --fix none --num-cifar10 50000 --num-imagenet 75000 \
    --ckpt-name cifar50_imagenet75_cond_ema \
    --pretrained-ckpt ./logs/2023-04-06/cifar10-imagenet/cifar50000_imagenet75000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_499_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-06 --fix none --num-cifar10 50000 --num-imagenet 100000 \
    --ckpt-name cifar50_imagenet100_cond_ema \
    --pretrained-ckpt ./logs/2023-04-06/cifar10-imagenet/cifar50000_imagenet100000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_424_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-06 --fix none --num-cifar10 50000 --num-imagenet 125000 \
    --ckpt-name cifar50_imagenet125_cond_ema \
    --pretrained-ckpt ./logs/2023-04-06/cifar10-imagenet/cifar50000_imagenet125000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_349_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-06 --fix none --num-cifar10 50000 --num-imagenet 150000 \
    --ckpt-name cifar50_imagenet150_cond_ema \
    --pretrained-ckpt ./logs/2023-04-06/cifar10-imagenet/cifar50000_imagenet150000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_324_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-06 --fix none --num-cifar10 50000 --num-imagenet 175000 \
    --ckpt-name cifar50_imagenet175_cond_ema \
    --pretrained-ckpt ./logs/2023-04-06/cifar10-imagenet/cifar50000_imagenet175000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_274_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-06 --fix none --num-cifar10 50000 --num-imagenet 200000 \
    --ckpt-name cifar50_imagenet200_cond_ema \
    --pretrained-ckpt ./logs/2023-04-06/cifar10-imagenet/cifar50000_imagenet200000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_249_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

# run 2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-07 --fix none --num-cifar10 50000 --num-imagenet 25000 \
    --ckpt-name cifar50_imagenet25_cond_ema \
    --pretrained-ckpt ./logs/2023-04-07/cifar10-imagenet/cifar50000_imagenet25000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_849_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-07 --fix none --num-cifar10 50000 --num-imagenet 50000 \
    --ckpt-name cifar50_imagenet50_cond_ema \
    --pretrained-ckpt ./logs/2023-04-07/cifar10-imagenet/cifar50000_imagenet50000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_624_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-07 --fix none --num-cifar10 50000 --num-imagenet 75000 \
    --ckpt-name cifar50_imagenet75_cond_ema \
    --pretrained-ckpt ./logs/2023-04-07/cifar10-imagenet/cifar50000_imagenet75000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_499_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-07 --fix none --num-cifar10 50000 --num-imagenet 100000 \
    --ckpt-name cifar50_imagenet100_cond_ema \
    --pretrained-ckpt ./logs/2023-04-07/cifar10-imagenet/cifar50000_imagenet100000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_424_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-07 --fix none --num-cifar10 50000 --num-imagenet 125000 \
    --ckpt-name cifar50_imagenet125_cond_ema \
    --pretrained-ckpt ./logs/2023-04-07/cifar10-imagenet/cifar50000_imagenet125000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_349_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-07 --fix none --num-cifar10 50000 --num-imagenet 150000 \
    --ckpt-name cifar50_imagenet150_cond_ema \
    --pretrained-ckpt ./logs/2023-04-07/cifar10-imagenet/cifar50000_imagenet150000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_324_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-07 --fix none --num-cifar10 50000 --num-imagenet 175000 \
    --ckpt-name cifar50_imagenet175_cond_ema \
    --pretrained-ckpt ./logs/2023-04-07/cifar10-imagenet/cifar50000_imagenet175000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_274_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-07 --fix none --num-cifar10 50000 --num-imagenet 200000 \
    --ckpt-name cifar50_imagenet200_cond_ema \
    --pretrained-ckpt ./logs/2023-04-07/cifar10-imagenet/cifar50000_imagenet200000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_249_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth
