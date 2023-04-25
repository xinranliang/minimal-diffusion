# run 1
CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNetSmall --dataset mnist --batch-size 100 --num-sampled-images 700 --guidance --sampling-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-08 --flip-left 0.5 --flip-right 0.5 \
    --class-cond --class-cond-dropout 0.1 --diffusion-steps 1000 \
    --ckpt-name mnist_cond_left0.5_right0.5_puncond0.1_epoch100_ema \
    --pretrained-ckpt ./logs/2023-04-08/mnist/left0.5_right0.5/UNetSmall_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_99_ema_0.9995.pth


# run 2
CUDA_VISIBLE_DEVICES=0 python main.py \
    --arch UNetSmall --dataset mnist --batch-size 100 --num-sampled-images 700 --guidance --sampling-only --sampling-steps 250 \
    --data-dir ./datasets --save-dir ./logs/ --date 2023-04-09 --flip-left 0.5 --flip-right 0.5 \
    --class-cond --class-cond-dropout 0.1 --diffusion-steps 1000 \
    --ckpt-name mnist_cond_left0.5_right0.5_puncond0.1_epoch100_ema \
    --pretrained-ckpt ./logs/2023-04-09/mnist/left0.5_right0.5/UNetSmall_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_99_ema_0.9995.pth
