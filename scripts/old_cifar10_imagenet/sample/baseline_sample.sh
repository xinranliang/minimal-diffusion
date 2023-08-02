# run 1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-06 --fix none --num-baseline 50000 \
    --ckpt-name baseline_mix50_cond_ema \
    --pretrained-ckpt ./logs/2023-04-06/mix-cifar10-imagenet/baseline_num50000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-06 --fix none --num-baseline 75000 \
    --ckpt-name baseline_mix75_cond_ema \
    --pretrained-ckpt ./logs/2023-04-06/mix-cifar10-imagenet/baseline_num75000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-06 --fix none --num-baseline 100000 \
    --ckpt-name baseline_mix100_cond_ema \
    --pretrained-ckpt ./logs/2023-04-06/mix-cifar10-imagenet/baseline_num100000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-06 --fix none --num-baseline 125000 \
    --ckpt-name baseline_mix125_cond_ema \
    --pretrained-ckpt ./logs/2023-04-06/mix-cifar10-imagenet/baseline_num125000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-06 --fix none --num-baseline 150000 \
    --ckpt-name baseline_mix150_cond_ema \
    --pretrained-ckpt ./logs/2023-04-06/mix-cifar10-imagenet/baseline_num150000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-06 --fix none --num-baseline 175000 \
    --ckpt-name baseline_mix175_cond_ema \
    --pretrained-ckpt ./logs/2023-04-06/mix-cifar10-imagenet/baseline_num175000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-06 --fix none --num-baseline 200000 \
    --ckpt-name baseline_mix200_cond_ema \
    --pretrained-ckpt ./logs/2023-04-06/mix-cifar10-imagenet/baseline_num200000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-06 --fix none --num-baseline 225000 \
    --ckpt-name baseline_mix225_cond_ema \
    --pretrained-ckpt ./logs/2023-04-06/mix-cifar10-imagenet/baseline_num225000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-06 --fix none --num-baseline 250000 \
    --ckpt-name baseline_mix250_cond_ema \
    --pretrained-ckpt ./logs/2023-04-06/mix-cifar10-imagenet/baseline_num250000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth


# run 2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-07 --fix none --num-baseline 50000 \
    --ckpt-name baseline_mix50_cond_ema \
    --pretrained-ckpt ./logs/2023-04-07/mix-cifar10-imagenet/baseline_num50000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_1249_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-07 --fix none --num-baseline 75000 \
    --ckpt-name baseline_mix75_cond_ema \
    --pretrained-ckpt ./logs/2023-04-07/mix-cifar10-imagenet/baseline_num75000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_849_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-07 --fix none --num-baseline 100000 \
    --ckpt-name baseline_mix100_cond_ema \
    --pretrained-ckpt ./logs/2023-04-07/mix-cifar10-imagenet/baseline_num100000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_624_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-07 --fix none --num-baseline 125000 \
    --ckpt-name baseline_mix125_cond_ema \
    --pretrained-ckpt ./logs/2023-04-07/mix-cifar10-imagenet/baseline_num125000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-07 --fix none --num-baseline 150000 \
    --ckpt-name baseline_mix150_cond_ema \
    --pretrained-ckpt ./logs/2023-04-07/mix-cifar10-imagenet/baseline_num150000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_350_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-07 --fix none --num-baseline 175000 \
    --ckpt-name baseline_mix175_cond_ema \
    --pretrained-ckpt ./logs/2023-04-07/mix-cifar10-imagenet/baseline_num175000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_349_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-07 --fix none --num-baseline 200000 \
    --ckpt-name baseline_mix200_cond_ema \
    --pretrained-ckpt ./logs/2023-04-07/mix-cifar10-imagenet/baseline_num200000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_324_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-07 --fix none --num-baseline 225000 \
    --ckpt-name baseline_mix225_cond_ema \
    --pretrained-ckpt ./logs/2023-04-07/mix-cifar10-imagenet/baseline_num225000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_274_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 8103 main.py \
    --arch UNet --dataset mix-cifar10-imagenet --batch-size 5000 --num-sampled-images 50000 --sampling-cifar-only --sampling-steps 250 \
    --data-dir ./datasets --class-cond --classifier-free-w 0.0 --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-07 --fix none --num-baseline 250000 \
    --ckpt-name baseline_mix250_cond_ema \
    --pretrained-ckpt ./logs/2023-04-07/mix-cifar10-imagenet/baseline_num250000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_249_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/cifar10-imagenet/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

