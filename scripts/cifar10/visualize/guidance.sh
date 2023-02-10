python visualize/visual_guidance.py --dataset cifar10 --date 2023-02-06 \
--train-color 1.0 --train-gray 0.0 --num-visualize 100 --num-classes 10 --num-samples 200 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1 \
--sample-config cifar10_cond_color1.0_gray0.0_puncond0.1_epoch1k_ema_num200

python visualize/visual_guidance.py --dataset cifar10 --date 2023-02-06 \
--train-color 1.0 --train-gray 0.0 --num-visualize 100 --num-classes 10 --num-samples 200 \
--diffusion-config UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.2 \
--sample-config cifar10_cond_color1.0_gray0.0_puncond0.2_epoch1k_ema_num200