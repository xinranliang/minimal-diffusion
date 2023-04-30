# run 1

# w.r.t 10k samples, color mode

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_color10000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor10_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_color20000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor20_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_color30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor30_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_color45000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor45_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_color60000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor60_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_color90000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor90_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_color120000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor120_ema_num10000_guidance0.0.npz

# w.r.t 10k samples, gray mode

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_gray10000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray10_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_gray20000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray20_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_gray30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray30_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_gray45000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray45_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_gray60000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray60_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_gray90000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray90_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_gray120000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray120_ema_num10000_guidance0.0.npz

# w.r.t 120k samples, color mode

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_color10000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor10_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_color20000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor20_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_color30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor30_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_color45000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor45_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_color60000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor60_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_color90000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor90_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_color120000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor120_ema_num120000_guidance0.0.npz

# w.r.t 120k samples, gray mode

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_gray10000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray10_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_gray20000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray20_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_gray30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray30_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_gray45000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray45_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_gray60000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray60_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_gray90000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray90_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half_gray120000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray120_ema_num120000_guidance0.0.npz



# run 2

# w.r.t 10k samples, color mode

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_color10000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor10_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_color20000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor20_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_color30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor30_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_color45000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor45_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_color60000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor60_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_color90000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor90_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_color120000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor120_ema_num10000_guidance0.0.npz

# w.r.t 10k samples, gray mode

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_gray10000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray10_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_gray20000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray20_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_gray30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray30_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_gray45000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray45_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_gray60000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray60_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_gray90000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray90_ema_num10000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_gray120000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray120_ema_num10000_guidance0.0.npz

# w.r.t 120k samples, color mode

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_color10000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor10_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_color20000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor20_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_color30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor30_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_color45000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor45_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_color60000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor60_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_color90000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor90_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_color120000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfcolor120_ema_num120000_guidance0.0.npz

# w.r.t 120k samples, gray mode

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_gray10000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray10_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_gray20000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray20_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_gray30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray30_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_gray45000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray45_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_gray60000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray60_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_gray90000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray90_ema_num120000_guidance0.0.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half_gray120000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_halfgray120_ema_num120000_guidance0.0.npz
