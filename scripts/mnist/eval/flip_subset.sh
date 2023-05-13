# run 1

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNetSmall --dataset mnist-subset \
    --data-dir ./datasets --diffusion-steps 1000 --sampling-steps 250 --num-sampled-images 50000 --batch-size 5000 --guidance \
    --class-cond --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-08 --flip-left 0.1 --flip-right 0.9 \
    --ckpt-name mnist_cond_left0.1_right0.9_puncond0.1_epoch100_ema \
    --pretrained-ckpt ./logs/2023-04-08/mnist-subset/left0.1_right0.9/UNetSmall_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_99_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/mnist-subset/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNetSmall --dataset mnist-subset \
    --data-dir ./datasets --diffusion-steps 1000 --sampling-steps 250 --num-sampled-images 50000 --batch-size 5000 --guidance \
    --class-cond --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-08 --flip-left 0.3 --flip-right 0.7 \
    --ckpt-name mnist_cond_left0.3_right0.7_puncond0.1_epoch100_ema \
    --pretrained-ckpt ./logs/2023-04-08/mnist-subset/left0.3_right0.7/UNetSmall_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_99_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/mnist-subset/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNetSmall --dataset mnist-subset \
    --data-dir ./datasets --diffusion-steps 1000 --sampling-steps 250 --num-sampled-images 50000 --batch-size 5000 --guidance \
    --class-cond --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-08 --flip-left 0.5 --flip-right 0.5 \
    --ckpt-name mnist_cond_left0.5_right0.5_puncond0.1_epoch100_ema \
    --pretrained-ckpt ./logs/2023-04-08/mnist-subset/left0.5_right0.5/UNetSmall_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_99_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/mnist-subset/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNetSmall --dataset mnist-subset \
    --data-dir ./datasets --diffusion-steps 1000 --sampling-steps 250 --num-sampled-images 50000 --batch-size 5000 --guidance \
    --class-cond --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-08 --flip-left 0.7 --flip-right 0.3 \
    --ckpt-name mnist_cond_left0.7_right0.3_puncond0.1_epoch100_ema \
    --pretrained-ckpt ./logs/2023-04-08/mnist-subset/left0.7_right0.3/UNetSmall_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_99_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/mnist-subset/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNetSmall --dataset mnist-subset \
    --data-dir ./datasets --diffusion-steps 1000 --sampling-steps 250 --num-sampled-images 50000 --batch-size 5000 --guidance \
    --class-cond --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-08 --flip-left 0.9 --flip-right 0.1 \
    --ckpt-name mnist_cond_left0.9_right0.1_puncond0.1_epoch100_ema \
    --pretrained-ckpt ./logs/2023-04-08/mnist-subset/left0.9_right0.1/UNetSmall_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_99_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/mnist-subset/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth


# run 2

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNetSmall --dataset mnist-subset \
    --data-dir ./datasets --diffusion-steps 1000 --sampling-steps 250 --num-sampled-images 50000 --batch-size 5000 --guidance \
    --class-cond --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-09 --flip-left 0.1 --flip-right 0.9 \
    --ckpt-name mnist_cond_left0.1_right0.9_puncond0.1_epoch100_ema \
    --pretrained-ckpt ./logs/2023-04-09/mnist-subset/left0.1_right0.9/UNetSmall_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_99_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/mnist-subset/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNetSmall --dataset mnist-subset \
    --data-dir ./datasets --diffusion-steps 1000 --sampling-steps 250 --num-sampled-images 50000 --batch-size 5000 --guidance \
    --class-cond --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-09 --flip-left 0.3 --flip-right 0.7 \
    --ckpt-name mnist_cond_left0.3_right0.7_puncond0.1_epoch100_ema \
    --pretrained-ckpt ./logs/2023-04-09/mnist-subset/left0.3_right0.7/UNetSmall_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_99_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/mnist-subset/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNetSmall --dataset mnist-subset \
    --data-dir ./datasets --diffusion-steps 1000 --sampling-steps 250 --num-sampled-images 50000 --batch-size 5000 --guidance \
    --class-cond --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-09 --flip-left 0.5 --flip-right 0.5 \
    --ckpt-name mnist_cond_left0.5_right0.5_puncond0.1_epoch100_ema \
    --pretrained-ckpt ./logs/2023-04-09/mnist-subset/left0.5_right0.5/UNetSmall_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_99_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/mnist-subset/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNetSmall --dataset mnist-subset \
    --data-dir ./datasets --diffusion-steps 1000 --sampling-steps 250 --num-sampled-images 50000 --batch-size 5000 --guidance \
    --class-cond --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-09 --flip-left 0.7 --flip-right 0.3 \
    --ckpt-name mnist_cond_left0.7_right0.3_puncond0.1_epoch100_ema \
    --pretrained-ckpt ./logs/2023-04-09/mnist-subset/left0.7_right0.3/UNetSmall_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_99_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/mnist-subset/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth

CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNetSmall --dataset mnist-subset \
    --data-dir ./datasets --diffusion-steps 1000 --sampling-steps 250 --num-sampled-images 50000 --batch-size 5000 --guidance \
    --class-cond --class-cond-dropout 0.1 \
    --save-dir ./logs/ --date 2023-04-09 --flip-left 0.9 --flip-right 0.1 \
    --ckpt-name mnist_cond_left0.9_right0.1_puncond0.1_epoch100_ema \
    --pretrained-ckpt ./logs/2023-04-09/mnist-subset/left0.9_right0.1/UNetSmall_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_99_ema_0.9995.pth \
    --domain-classifier ./logs/2023-04-08/mnist-subset/domain_classifier/bs256_lr0.001_decay0.0001/ckpt/model_param_final.pth
