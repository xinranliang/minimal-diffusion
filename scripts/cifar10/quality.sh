echo "Evaluating FID score: Training with 0.99 color and 0.01 gray; Sample color"
python metrics/fid_score.py --true ./datasets/cifar10/train_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.99_gray0.01/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.99_gray0.01_epoch_950_num50000_color_images.npy \
--gpu 0 --batch-size 500
# FID: 7.73 (0.073)

echo "Evaluating FID score: Training with 0.95 color and 0.05 gray; Sample color"
python metrics/fid_score.py --true ./datasets/cifar10/train_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.95_gray0.05_epoch_950_num50000_color_images.npy \
--gpu 0 --batch-size 500
# FID: 6.31 (0.037)

echo "Evaluating FID score: Training with 0.95 color and 0.05 gray; Sample gray"
python metrics/fid_score.py --true ./datasets/cifar10/test_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.95_gray0.05_epoch_950_num10000_gray_images.npy \
--gpu 0 --batch-size 500
# FID: 10.48 (0.105)

echo "Evaluating FID score: Training with 0.05 color and 0.95 gray; Sample gray"
python metrics/fid_score.py --true ./datasets/cifar10/train_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.05_gray0.95_epoch_950_num50000_gray_images.npy \
--gpu 0 --batch-size 500
# FID: 4.39 (0.048)

echo "Evaluating FID score: Training with 0.05 color and 0.95 gray; Sample color"
python metrics/fid_score.py --true ./datasets/cifar10/test_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.05_gray0.95_epoch_950_num10000_color_images.npy \
--gpu 0 --batch-size 500
# FID: 17.77 (0.167)

echo "Evaluating FID score: Training with 0.01 color and 0.99 gray; Sample gray"
python metrics/fid_score.py --true ./datasets/cifar10/train_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.01_gray0.99/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.01_gray0.99_epoch_950_num50000_gray_images.npy \
--gpu 0 --batch-size 500
# FID: 4.73 (0.065)