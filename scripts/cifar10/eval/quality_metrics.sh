echo "Evaluating FID score: Training with 1.0 color and 0.0 gray; Sample color"
python metrics/fid_score.py --true ./datasets/cifar10/train_color.npy \
--fake ./logs/2022-12-08/cifar10/color1.0_gray0.0/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color1.0_gray0.0_epoch_950_num50000_images.npy \
--gpu 0 --batch-size 500
# FID: 14.73 (0.185)

echo "Evaluating FID score: Training with 0.99 color and 0.01 gray; Sample color"
python metrics/fid_score.py --true ./datasets/cifar10/train_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.99_gray0.01/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.99_gray0.01_epoch_950_num50000_color_images.npy \
--gpu 0 --batch-size 500
# FID: 13.78 (0.326)

echo "Evaluating FID score: Training with 0.99 color and 0.01 gray; Sample gray"
python metrics/fid_score.py --true ./datasets/cifar10/train_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.99_gray0.01/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.99_gray0.01_epoch_950_num10000_gray_images.npy \
--gpu 0 --batch-size 500
# FID: 23.89 (0.536)

echo "Evaluating FID score: Training with 0.95 color and 0.05 gray; Sample color"
python metrics/fid_score.py --true ./datasets/cifar10/train_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.95_gray0.05_epoch_950_num50000_color_images.npy \
--gpu 0 --batch-size 500
# FID: 12.38 (0.225)

echo "Evaluating FID score: Training with 0.95 color and 0.05 gray; Sample gray"
python metrics/fid_score.py --true ./datasets/cifar10/train_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.95_gray0.05_epoch_950_num10000_gray_images.npy \
--gpu 0 --batch-size 500
# FID: 10.31 (0.081)

echo "Evaluating FID score: Training with 0.05 color and 0.95 gray; Sample gray"
python metrics/fid_score.py --true ./datasets/cifar10/train_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.05_gray0.95_epoch_950_num50000_gray_images.npy \
--gpu 0 --batch-size 500
# FID: 9.25 (0.180)

echo "Evaluating FID score: Training with 0.05 color and 0.95 gray; Sample color"
python metrics/fid_score.py --true ./datasets/cifar10/train_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.05_gray0.95_epoch_950_num10000_color_images.npy \
--gpu 0 --batch-size 500
# FID: 17.64 (0.409)

echo "Evaluating FID score: Training with 0.01 color and 0.99 gray; Sample gray"
python metrics/fid_score.py --true ./datasets/cifar10/train_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.01_gray0.99/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.01_gray0.99_epoch_950_num50000_gray_images.npy \
--gpu 0 --batch-size 500
# FID: 9.54 (0.152)

echo "Evaluating FID score: Training with 0.01 color and 0.99 gray; Sample color"
python metrics/fid_score.py --true ./datasets/cifar10/train_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.01_gray0.99/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.01_gray0.99_epoch_950_num10000_color_images.npy \
--gpu 0 --batch-size 500
# FID: 21.29 (0.381)

echo "Evaluating FID score: Training with 0.0 color and 1.0 gray; Sample gray"
python metrics/fid_score.py --true ./datasets/cifar10/train_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.0_gray1.0/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.0_gray1.0_epoch_950_num50000_images.npy \
--gpu 0 --batch-size 500
# FID: 8.71 (0.135)


echo "Evaluating KID score: Training with 1.0 color and 0.0 gray; Sample color"
python metrics/kid_score.py --true ./datasets/cifar10/train_color.npy \
--fake ./logs/2022-12-08/cifar10/color1.0_gray0.0/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color1.0_gray0.0_epoch_950_num50000_images.npy \
--gpu 0 --batch-size 500
# KID: 0.006076

echo "Evaluating KID score: Training with 0.99 color and 0.01 gray; Sample color"
python metrics/kid_score.py --true ./datasets/cifar10/train_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.99_gray0.01/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.99_gray0.01_epoch_950_num50000_color_images.npy \
--gpu 0 --batch-size 500
# KID: 0.00543

echo "Evaluating KID score: Training with 0.99 color and 0.01 gray; Sample gray"
python metrics/kid_score.py --true ./datasets/cifar10/train_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.99_gray0.01/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.99_gray0.01_epoch_950_num10000_gray_images.npy \
--gpu 0 --batch-size 500
# KID: 0.0134

echo "Evaluating KID score: Training with 0.95 color and 0.05 gray; Sample color"
python metrics/kid_score.py --true ./datasets/cifar10/train_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.95_gray0.05_epoch_950_num50000_color_images.npy \
--gpu 0 --batch-size 500
# KID: 0.00519

echo "Evaluating KID score: Training with 0.95 color and 0.05 gray; Sample gray"
python metrics/kid_score.py --true ./datasets/cifar10/train_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.95_gray0.05_epoch_950_num10000_gray_images.npy \
--gpu 0 --batch-size 500
# KID: 0.00221

echo "Evaluating KID score: Training with 0.05 color and 0.95 gray; Sample gray"
python metrics/kid_score.py --true ./datasets/cifar10/train_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.05_gray0.95_epoch_950_num50000_gray_images.npy \
--gpu 0 --batch-size 500
# KID: 0.002218

echo "Evaluating KID score: Training with 0.05 color and 0.95 gray; Sample color"
python metrics/kid_score.py --true ./datasets/cifar10/train_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.05_gray0.95_epoch_950_num10000_color_images.npy \
--gpu 0 --batch-size 500
# KID: 0.007674

echo "Evaluating KID score: Training with 0.01 color and 0.99 gray; Sample gray"
python metrics/kid_score.py --true ./datasets/cifar10/train_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.01_gray0.99/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.01_gray0.99_epoch_950_num50000_gray_images.npy \
--gpu 0 --batch-size 500
# KID: 0.002347

echo "Evaluating KID score: Training with 0.01 color and 0.99 gray; Sample color"
python metrics/kid_score.py --true ./datasets/cifar10/train_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.01_gray0.99/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.01_gray0.99_epoch_950_num10000_color_images.npy \
--gpu 0 --batch-size 500
# KID: 0.012087

echo "Evaluating KID score: Training with 0.0 color and 1.0 gray; Sample gray"
python metrics/kid_score.py --true ./datasets/cifar10/train_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.0_gray1.0/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.0_gray1.0_epoch_950_num50000_images.npy \
--gpu 0 --batch-size 500
# KID: 0.002472