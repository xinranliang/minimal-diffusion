# all color cond

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

# 2500 C + 0 G

python visualize/visual_inspect.py --dataset cifar10 --date 2023-02-07 \
--train-color 2500 --train-gray 0 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1 \
--sample-config cifar10_cond_color25_gray0_ema_num50000_guidance0.0 \
--class-cond --num-classes 10

python visualize/visual_inspect.py --dataset cifar10 --date 2023-02-07 \
--train-color 2500 --train-gray 0 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1 \
--sample-config cifar10_cond_color25_gray0_ema_num50000_guidance0.5 \
--class-cond --num-classes 10

python visualize/visual_inspect.py --dataset cifar10 --date 2023-02-07 \
--train-color 2500 --train-gray 0 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1 \
--sample-config cifar10_cond_color25_gray0_ema_num50000_guidance1.0 \
--class-cond --num-classes 10

python visualize/visual_inspect.py --dataset cifar10 --date 2023-02-07 \
--train-color 2500 --train-gray 0 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1 \
--sample-config cifar10_cond_color25_gray0_ema_num50000_guidance2.0 \
--class-cond --num-classes 10

python visualize/visual_inspect.py --dataset cifar10 --date 2023-02-07 \
--train-color 2500 --train-gray 0 --num-visualize 100 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_4096_dropprob_0.1 \
--sample-config cifar10_cond_color25_gray0_ema_num50000_guidance4.0 \
--class-cond --num-classes 10