echo "Evaluating FID score from training distribution with 0.99 colored and 0.01 grayscaled"
python metrics/fid_score.py --true ./datasets/cifar10/real_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.99_gray0.01/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.99_gray0.01_epoch_950_num50000_images.npy \
--gpu 0 --batch-size 50

echo "Evaluating FID score from training distribution with 0.95 colored and 0.05 grayscaled"
python metrics/fid_score.py --true ./datasets/cifar10/real_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.95_gray0.05_epoch_950_num50000_images.npy \
--gpu 0 --batch-size 50

echo "Evaluating FID score from training distribution with 0.05 colored and 0.95 grayscaled"
python metrics/fid_score.py --true ./datasets/cifar10/real_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.05_gray0.95_epoch_950_num50000_images.npy \
--gpu 0 --batch-size 50

echo "Evaluating FID score from training distribution with 0.01 colored and 0.99 grayscaled"
python metrics/fid_score.py --true ./datasets/cifar10/real_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.01_gray0.99/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.01_gray0.99_epoch_950_num50000_images.npy \
--gpu 0 --batch-size 50