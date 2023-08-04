

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 guidance.py \
    --arch UNet --dataset fairface --epochs 1000 --batch-size 2500 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --save-dir ./logs/ --date 2023-07-25 --female-ratio 0.1 --male-ratio 0.9 \
    --class-cond --class-cond-dropout 0.1 --ema_w 0.9995 --ckpt-sample-freq 100 \
    --sampling-only --num-sampled-images 50000 \
    --ckpt-name fairface_gendersubset_f0.1_m0.9_ema \
    --pretrained-ckpt ./logs/2023-07-25/fairface/female0.1_male0.9/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_128_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth 

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 guidance.py \
    --arch UNet --dataset fairface --epochs 1000 --batch-size 2500 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --save-dir ./logs/ --date 2023-07-25 --female-ratio 0.3 --male-ratio 0.7 \
    --class-cond --class-cond-dropout 0.1 --ema_w 0.9995 --ckpt-sample-freq 100 \
    --sampling-only --num-sampled-images 50000 \
    --ckpt-name fairface_gendersubset_f0.3_m0.7_ema \
    --pretrained-ckpt ./logs/2023-07-25/fairface/female0.3_male0.7/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_128_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth 

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 guidance.py \
    --arch UNet --dataset fairface --epochs 1000 --batch-size 2500 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --save-dir ./logs/ --date 2023-07-25 --female-ratio 0.5 --male-ratio 0.5 \
    --class-cond --class-cond-dropout 0.1 --ema_w 0.9995 --ckpt-sample-freq 100 \
    --sampling-only --num-sampled-images 50000 \
    --ckpt-name fairface_gendersubset_f0.5_m0.5_ema \
    --pretrained-ckpt ./logs/2023-07-25/fairface/female0.5_male0.5/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_128_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth 

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 guidance.py \
    --arch UNet --dataset fairface --epochs 1000 --batch-size 2500 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --save-dir ./logs/ --date 2023-07-25 --female-ratio 0.7 --male-ratio 0.3 \
    --class-cond --class-cond-dropout 0.1 --ema_w 0.9995 --ckpt-sample-freq 100 \
    --sampling-only --num-sampled-images 50000 \
    --ckpt-name fairface_gendersubset_f0.7_m0.3_ema \
    --pretrained-ckpt ./logs/2023-07-25/fairface/female0.7_male0.3/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_128_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth 

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 guidance.py \
    --arch UNet --dataset fairface --epochs 1000 --batch-size 2500 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --save-dir ./logs/ --date 2023-07-25 --female-ratio 0.9 --male-ratio 0.1 \
    --class-cond --class-cond-dropout 0.1 --ema_w 0.9995 --ckpt-sample-freq 100 \
    --sampling-only --num-sampled-images 50000 \
    --ckpt-name fairface_gendersubset_f0.9_m0.1_ema \
    --pretrained-ckpt ./logs/2023-07-25/fairface/female0.9_male0.1/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_128_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth 


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 guidance.py \
    --arch UNet --dataset fairface --epochs 1000 --batch-size 2500 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --save-dir ./logs/ --date 2023-07-26 --female-ratio 0.1 --male-ratio 0.9 \
    --class-cond --class-cond-dropout 0.1 --ema_w 0.9995 --ckpt-sample-freq 100 \
    --sampling-only --num-sampled-images 50000 \
    --ckpt-name fairface_gendersubset_f0.1_m0.9_ema \
    --pretrained-ckpt ./logs/2023-07-26/fairface/female0.1_male0.9/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_128_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth 

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 guidance.py \
    --arch UNet --dataset fairface --epochs 1000 --batch-size 2500 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --save-dir ./logs/ --date 2023-07-26 --female-ratio 0.3 --male-ratio 0.7 \
    --class-cond --class-cond-dropout 0.1 --ema_w 0.9995 --ckpt-sample-freq 100 \
    --sampling-only --num-sampled-images 50000 \
    --ckpt-name fairface_gendersubset_f0.3_m0.7_ema \
    --pretrained-ckpt ./logs/2023-07-26/fairface/female0.3_male0.7/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_128_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth 

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 guidance.py \
    --arch UNet --dataset fairface --epochs 1000 --batch-size 2500 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --save-dir ./logs/ --date 2023-07-26 --female-ratio 0.5 --male-ratio 0.5 \
    --class-cond --class-cond-dropout 0.1 --ema_w 0.9995 --ckpt-sample-freq 100 \
    --sampling-only --num-sampled-images 50000 \
    --ckpt-name fairface_gendersubset_f0.5_m0.5_ema \
    --pretrained-ckpt ./logs/2023-07-26/fairface/female0.5_male0.5/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_128_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth 

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 guidance.py \
    --arch UNet --dataset fairface --epochs 1000 --batch-size 2500 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --save-dir ./logs/ --date 2023-07-26 --female-ratio 0.7 --male-ratio 0.3 \
    --class-cond --class-cond-dropout 0.1 --ema_w 0.9995 --ckpt-sample-freq 100 \
    --sampling-only --num-sampled-images 50000 \
    --ckpt-name fairface_gendersubset_f0.7_m0.3_ema \
    --pretrained-ckpt ./logs/2023-07-26/fairface/female0.7_male0.3/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_128_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth 

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8103 guidance.py \
    --arch UNet --dataset fairface --epochs 1000 --batch-size 2500 --lr 1e-4 --sampling-steps 250 \
    --data-dir ./datasets --diffusion-steps 1000 --save-dir ./logs/ --date 2023-07-26 --female-ratio 0.9 --male-ratio 0.1 \
    --class-cond --class-cond-dropout 0.1 --ema_w 0.9995 --ckpt-sample-freq 100 \
    --sampling-only --num-sampled-images 50000 \
    --ckpt-name fairface_gendersubset_f0.9_m0.1_ema \
    --pretrained-ckpt ./logs/2023-07-26/fairface/female0.9_male0.1/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_128_dropprob_0.1/ckpt/epoch_final_ema_0.9995.pth 
