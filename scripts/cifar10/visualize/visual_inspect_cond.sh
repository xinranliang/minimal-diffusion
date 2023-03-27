# all color cond with guidance

python visualize/visual_inspect.py --dataset cifar10 --date 2023-02-06 \
--train-color 1.0 --train-gray 0.0 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1 \
--sample-config cifar10_cond_color1.0_gray0.0_puncond0.1_epoch1k_ema_num50000_guidance0.0 \
--class-cond --num-classes 10

python visualize/visual_inspect.py --dataset cifar10 --date 2023-02-06 \
--train-color 1.0 --train-gray 0.0 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1 \
--sample-config cifar10_cond_color1.0_gray0.0_puncond0.1_epoch1k_ema_num50000_guidance0.5 \
--class-cond --num-classes 10

python visualize/visual_inspect.py --dataset cifar10 --date 2023-02-06 \
--train-color 1.0 --train-gray 0.0 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1 \
--sample-config cifar10_cond_color1.0_gray0.0_puncond0.1_epoch1k_ema_num50000_guidance1.0 \
--class-cond --num-classes 10

python visualize/visual_inspect.py --dataset cifar10 --date 2023-02-06 \
--train-color 1.0 --train-gray 0.0 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1 \
--sample-config cifar10_cond_color1.0_gray0.0_puncond0.1_epoch1k_ema_num50000_guidance2.0 \
--class-cond --num-classes 10

python visualize/visual_inspect.py --dataset cifar10 --date 2023-02-06 \
--train-color 1.0 --train-gray 0.0 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1 \
--sample-config cifar10_cond_color1.0_gray0.0_puncond0.1_epoch1k_ema_num50000_guidance4.0 \
--class-cond --num-classes 10

# 0.5C + 0.5G

python visualize/visual_inspect.py --dataset cifar10 --date 2023-02-06 \
--train-color 0.5 --train-gray 0.5 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_10000_dropprob_0.1 \
--sample-config cifar10_cond_color0.5_gray0.5_puncond0.1_epoch1k_ema_num50000_guidance0.0 \
--class-cond --num-classes 10

python visualize/visual_inspect.py --dataset cifar10 --date 2023-02-06 \
--train-color 0.5 --train-gray 0.5 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_10000_dropprob_0.1 \
--sample-config cifar10_cond_color0.5_gray0.5_puncond0.1_epoch1k_ema_num50000_guidance0.2 \
--class-cond --num-classes 10

python visualize/visual_inspect.py --dataset cifar10 --date 2023-02-06 \
--train-color 0.5 --train-gray 0.5 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_10000_dropprob_0.1 \
--sample-config cifar10_cond_color0.5_gray0.5_puncond0.1_epoch1k_ema_num50000_guidance0.4 \
--class-cond --num-classes 10

python visualize/visual_inspect.py --dataset cifar10 --date 2023-02-06 \
--train-color 0.5 --train-gray 0.5 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_10000_dropprob_0.1 \
--sample-config cifar10_cond_color0.5_gray0.5_puncond0.1_epoch1k_ema_num50000_guidance0.6 \
--class-cond --num-classes 10

python visualize/visual_inspect.py --dataset cifar10 --date 2023-02-06 \
--train-color 0.5 --train-gray 0.5 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_10000_dropprob_0.1 \
--sample-config cifar10_cond_color0.5_gray0.5_puncond0.1_epoch1k_ema_num50000_guidance0.8 \
--class-cond --num-classes 10

python visualize/visual_inspect.py --dataset cifar10 --date 2023-02-06 \
--train-color 0.5 --train-gray 0.5 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_10000_dropprob_0.1 \
--sample-config cifar10_cond_color0.5_gray0.5_puncond0.1_epoch1k_ema_num50000_guidance1.0 \
--class-cond --num-classes 10

python visualize/visual_inspect.py --dataset cifar10 --date 2023-02-06 \
--train-color 0.5 --train-gray 0.5 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_10000_dropprob_0.1 \
--sample-config cifar10_cond_color0.5_gray0.5_puncond0.1_epoch1k_ema_num50000_guidance2.0 \
--class-cond --num-classes 10

python visualize/visual_inspect.py --dataset cifar10 --date 2023-02-06 \
--train-color 0.5 --train-gray 0.5 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_10000_dropprob_0.1 \
--sample-config cifar10_cond_color0.5_gray0.5_puncond0.1_epoch1k_ema_num50000_guidance3.0 \
--class-cond --num-classes 10

python visualize/visual_inspect.py --dataset cifar10 --date 2023-02-06 \
--train-color 0.5 --train-gray 0.5 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_10000_dropprob_0.1 \
--sample-config cifar10_cond_color0.5_gray0.5_puncond0.1_epoch1k_ema_num50000_guidance4.0 \
--class-cond --num-classes 10