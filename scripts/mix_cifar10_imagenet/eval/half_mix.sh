# run 1

# w.r.t 10k samples, color mode

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half10000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half10_ema_num10000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half20000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half20_ema_num10000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half30_ema_num10000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half45000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half45_ema_num10000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half60000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half60_ema_num10000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half90000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half90_ema_num10000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half120000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half120_ema_num10000_color.npz

# w.r.t 10k samples, gray mode

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half10000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half10_ema_num10000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half20000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half20_ema_num10000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half30_ema_num10000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half45000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half45_ema_num10000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half60000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half60_ema_num10000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half90000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half90_ema_num10000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 10000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half120000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half120_ema_num10000_gray.npz

# w.r.t 120k samples, color mode

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half10000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half10_ema_num120000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half20000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half20_ema_num120000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half30_ema_num120000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half45000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half45_ema_num120000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half60000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half60_ema_num120000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half90000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half90_ema_num120000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half120000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half120_ema_num120000_color.npz

# w.r.t 120k samples, gray mode

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half10000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half10_ema_num120000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half20000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half20_ema_num120000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half30_ema_num120000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half45000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half45_ema_num120000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half60000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half60_ema_num120000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half90000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half90_ema_num120000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-02 --num-samples 120000 \
--fake ./logs/2023-04-02/mix-cifar10-imagenet/half120000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half120_ema_num120000_gray.npz



# run 2

# w.r.t 10k samples, color mode

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half10000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half10_ema_num10000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half20000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half20_ema_num10000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half30_ema_num10000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half45000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half45_ema_num10000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half60000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half60_ema_num10000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half90000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half90_ema_num10000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half120000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half120_ema_num10000_color.npz

# w.r.t 10k samples, gray mode

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half10000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half10_ema_num10000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half20000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half20_ema_num10000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half30_ema_num10000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half45000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half45_ema_num10000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half60000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half60_ema_num10000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half90000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half90_ema_num10000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 10000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half120000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half120_ema_num10000_gray.npz

# w.r.t 120k samples, color mode

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half10000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half10_ema_num120000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half20000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half20_ema_num120000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half30_ema_num120000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half45000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half45_ema_num120000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half60000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half60_ema_num120000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half90000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half90_ema_num120000_color.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode color \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half120000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half120_ema_num120000_color.npz

# w.r.t 120k samples, gray mode

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half10000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half10_ema_num120000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half20000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half20_ema_num120000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half30000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half30_ema_num120000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half45000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half45_ema_num120000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half60000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half60_ema_num120000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half90000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half90_ema_num120000_gray.npz

python metrics-all/compute_metrics.py --dataset mix-cifar10-imagenet --resolution 32 --mode legacy_tensorflow --batch-size 1000 --num-gpus 1 --real-mode gray \
--date 2023-04-03 --num-samples 120000 \
--fake ./logs/2023-04-03/mix-cifar10-imagenet/half120000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/mix_cifar10_imagenet_cond_half120_ema_num120000_gray.npz
