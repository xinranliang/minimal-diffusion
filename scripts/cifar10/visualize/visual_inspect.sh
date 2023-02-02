# uncond color baseline
python visualize/visual_inspect.py --dataset cifar10 --date 2023-01-22 \
--train-color 1.0 --train-gray 0.0 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096 \
--sample-config cifar10_uncond_color1.0_gray0.0_epoch_1k_ema_num50000 

# cond color baseline
python visualize/visual_inspect.py --dataset cifar10 --date 2023-01-22 \
--train-color 1.0 --train-gray 0.0 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096 \
--sample-config cifar10_cond_color1.0_gray0.0_epoch_1k_ema_num50000 \
--class-cond --num-classes 10

# uncond gray baseline
python visualize/visual_inspect.py --dataset cifar10 --date 2023-01-22 \
--train-color 0.0 --train-gray 1.0 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096 \
--sample-config cifar10_cond_color0.0_gray1.0_epoch_1k_ema_num50000 

# cond gray baseline
python visualize/visual_inspect.py --dataset cifar10 --date 2023-01-22 \
--train-color 0.0 --train-gray 1.0 --num-visualize 200 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096 \
--sample-config cifar10_cond_color0.0_gray1.0_epoch_1k_ema_num50000 \
--class-cond --num-classes 10

# 95C-05G
# uncond color
python visualize/visual_inspect.py --dataset cifar10 --date 2023-01-22 \
--train-color 0.95 --train-gray 0.05 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096 \
--sample-config cifar10_uncond_color0.95_gray0.05_epoch1000_ema_num50000_color
# uncond gray
python visualize/visual_inspect.py --dataset cifar10 --date 2023-01-22 \
--train-color 0.95 --train-gray 0.05 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096 \
--sample-config cifar10_uncond_color0.95_gray0.05_epoch1000_ema_num50000_gray
# cond color
python visualize/visual_inspect.py --dataset cifar10 --date 2023-01-22 \
--train-color 0.95 --train-gray 0.05 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096 \
--sample-config cifar10_cond_color0.95_gray0.05_epoch1000_ema_num50000_color \
--class-cond --num-classes 10
# cond gray
python visualize/visual_inspect.py --dataset cifar10 --date 2023-01-22 \
--train-color 0.95 --train-gray 0.05 --num-visualize 200 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096 \
--sample-config cifar10_cond_color0.95_gray0.05_epoch1000_ema_num50000_gray \
--class-cond --num-classes 10

# 90C-10G
# uncond color
python visualize/visual_inspect.py --dataset cifar10 --date 2023-01-22 \
--train-color 0.9 --train-gray 0.1 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096 \
--sample-config cifar10_uncond_color0.9_gray0.1_epoch1000_ema_num50000_color
# uncond gray
python visualize/visual_inspect.py --dataset cifar10 --date 2023-01-22 \
--train-color 0.9 --train-gray 0.1 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096 \
--sample-config cifar10_uncond_color0.9_gray0.1_epoch1000_ema_num50000_gray
# cond color
python visualize/visual_inspect.py --dataset cifar10 --date 2023-01-22 \
--train-color 0.9 --train-gray 0.1 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096 \
--sample-config cifar10_cond_color0.9_gray0.1_epoch1000_ema_num50000_color \
--class-cond --num-classes 10
# cond gray
python visualize/visual_inspect.py --dataset cifar10 --date 2023-01-22 \
--train-color 0.9 --train-gray 0.1 --num-visualize 200 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096 \
--sample-config cifar10_cond_color0.9_gray0.1_epoch1000_ema_num50000_gray \
--class-cond --num-classes 10

# 10C-90G
# uncond color
python visualize/visual_inspect.py --dataset cifar10 --date 2023-01-22 \
--train-color 0.1 --train-gray 0.9 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096 \
--sample-config cifar10_uncond_color0.1_gray0.9_epoch1000_ema_num50000_color
# uncond gray
python visualize/visual_inspect.py --dataset cifar10 --date 2023-01-22 \
--train-color 0.1 --train-gray 0.9 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096 \
--sample-config cifar10_uncond_color0.1_gray0.9_epoch1000_ema_num50000_gray
# cond color
python visualize/visual_inspect.py --dataset cifar10 --date 2023-01-22 \
--train-color 0.1 --train-gray 0.9 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096 \
--sample-config cifar10_cond_color0.1_gray0.9_epoch1000_ema_num50000_color \
--class-cond --num-classes 10
# cond gray
python visualize/visual_inspect.py --dataset cifar10 --date 2023-01-22 \
--train-color 0.1 --train-gray 0.9 --num-visualize 200 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096 \
--sample-config cifar10_cond_color0.1_gray0.9_epoch1000_ema_num50000_gray \
--class-cond --num-classes 10

# 05C-90G
# uncond color
python visualize/visual_inspect.py --dataset cifar10 --date 2023-01-22 \
--train-color 0.05 --train-gray 0.95 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096 \
--sample-config cifar10_uncond_color0.05_gray0.95_epoch1000_ema_num50000_color
# uncond gray
python visualize/visual_inspect.py --dataset cifar10 --date 2023-01-22 \
--train-color 0.05 --train-gray 0.95 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_4096 \
--sample-config cifar10_uncond_color0.05_gray0.95_epoch1000_ema_num50000_gray
# cond color
python visualize/visual_inspect.py --dataset cifar10 --date 2023-01-22 \
--train-color 0.05 --train-gray 0.95 --num-visualize 200 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096 \
--sample-config cifar10_cond_color0.05_gray0.95_epoch1000_ema_num50000_color \
--class-cond --num-classes 10
# cond gray
python visualize/visual_inspect.py --dataset cifar10 --date 2023-01-22 \
--train-color 0.05 --train-gray 0.95 --num-visualize 200 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096 \
--sample-config cifar10_cond_color0.05_gray0.95_epoch1000_ema_num50000_gray \
--class-cond --num-classes 10