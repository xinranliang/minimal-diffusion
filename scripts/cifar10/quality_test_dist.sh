echo "Evaluating FID score: Training with 0.99 color and 0.01 gray; Sample color"
python metrics/fid_score.py --true ./datasets/cifar10/test_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.99_gray0.01/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.99_gray0.01_epoch_950_num50000_color_images.npy \
--gpu 0 --batch-size 500
# FID: 14.13 (0.171)

echo "Evaluating FID score: Training with 0.99 color and 0.01 gray; Sample gray"
python metrics/fid_score.py --true ./datasets/cifar10/test_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.99_gray0.01/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.99_gray0.01_epoch_950_num10000_gray_images.npy \
--gpu 0 --batch-size 500
# FID: 23.44 (0.423)

echo "Evaluating FID score: Training with 0.95 color and 0.05 gray; Sample color"
python metrics/fid_score.py --true ./datasets/cifar10/test_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.95_gray0.05_epoch_950_num50000_color_images.npy \
--gpu 0 --batch-size 500
# FID: 12.59 (0.137)

echo "Evaluating FID score: Training with 0.95 color and 0.05 gray; Sample gray"
python metrics/fid_score.py --true ./datasets/cifar10/test_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.95_gray0.05_epoch_950_num10000_gray_images.npy \
--gpu 0 --batch-size 500
# FID: 10.36 (0.191)

echo "Evaluating FID score: Training with 0.05 color and 0.95 gray; Sample gray"
python metrics/fid_score.py --true ./datasets/cifar10/test_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.05_gray0.95_epoch_950_num50000_gray_images.npy \
--gpu 0 --batch-size 500
# FID: 9.26 (0.080)

echo "Evaluating FID score: Training with 0.05 color and 0.95 gray; Sample color"
python metrics/fid_score.py --true ./datasets/cifar10/test_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.05_gray0.95_epoch_950_num10000_color_images.npy \
--gpu 0 --batch-size 500
# FID: 17.77 (0.217)

echo "Evaluating FID score: Training with 0.01 color and 0.99 gray; Sample gray"
python metrics/fid_score.py --true ./datasets/cifar10/test_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.01_gray0.99/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.01_gray0.99_epoch_950_num50000_gray_images.npy \
--gpu 0 --batch-size 500
# FID: 9.48 (0.145)

echo "Evaluating FID score: Training with 0.01 color and 0.99 gray; Sample color"
python metrics/fid_score.py --true ./datasets/cifar10/test_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.01_gray0.99/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.01_gray0.99_epoch_950_num10000_color_images.npy \
--gpu 0 --batch-size 500
# FID: 21.71 (0.309)


echo "Evaluating KID score: Training with 0.99 color and 0.01 gray; Sample color"
python metrics/kid_score.py --true ./datasets/cifar10/test_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.99_gray0.01/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.99_gray0.01_epoch_950_num50000_color_images.npy \
--gpu 0 --batch-size 500
# KID: 0.005737

echo "Evaluating KID score: Training with 0.99 color and 0.01 gray; Sample gray"
python metrics/kid_score.py --true ./datasets/cifar10/test_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.99_gray0.01/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.99_gray0.01_epoch_950_num10000_gray_images.npy \
--gpu 0 --batch-size 500
# KID: 0.013255

echo "Evaluating KID score: Training with 0.95 color and 0.05 gray; Sample color"
python metrics/kid_score.py --true ./datasets/cifar10/test_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.95_gray0.05_epoch_950_num50000_color_images.npy \
--gpu 0 --batch-size 500
# KID: 0.005496

echo "Evaluating KID score: Training with 0.95 color and 0.05 gray; Sample gray"
python metrics/kid_score.py --true ./datasets/cifar10/test_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.95_gray0.05_epoch_950_num10000_gray_images.npy \
--gpu 0 --batch-size 500
# KID: 0.002321

echo "Evaluating KID score: Training with 0.05 color and 0.95 gray; Sample gray"
python metrics/kid_score.py --true ./datasets/cifar10/test_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.05_gray0.95_epoch_950_num50000_gray_images.npy \
--gpu 0 --batch-size 500
# KID: 0.002109

echo "Evaluating KID score: Training with 0.05 color and 0.95 gray; Sample color"
python metrics/kid_score.py --true ./datasets/cifar10/test_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.05_gray0.95_epoch_950_num10000_color_images.npy \
--gpu 0 --batch-size 500
# KID: 0.007864

echo "Evaluating KID score: Training with 0.01 color and 0.99 gray; Sample gray"
python metrics/kid_score.py --true ./datasets/cifar10/test_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.01_gray0.99/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.01_gray0.99_epoch_950_num50000_gray_images.npy \
--gpu 0 --batch-size 500
# KID: 0.002404

echo "Evaluating KID score: Training with 0.01 color and 0.99 gray; Sample color"
python metrics/kid_score.py --true ./datasets/cifar10/test_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.01_gray0.99/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.01_gray0.99_epoch_950_num10000_color_images.npy \
--gpu 0 --batch-size 500
# KID: 0.012287