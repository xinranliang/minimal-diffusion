echo "Evaluating precision recall: Training with 1.0 color and 0.0 gray; Sample color"
python metrics/precision_recall.py --true ./datasets/cifar10/test_color.npy \
--fake ./logs/2022-12-08/cifar10/color1.0_gray0.0/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color1.0_gray0.0_epoch_950_num50000_images.npy \
--gpu 0 --batch-size 500
# precision:  0.7352
# recall:  0.5451

echo "Evaluating precision recall: Training with 0.99 color and 0.01 gray; Sample color"
python metrics/precision_recall.py --true ./datasets/cifar10/test_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.99_gray0.01/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.99_gray0.01_epoch_950_num50000_color_images.npy \
--gpu 0 --batch-size 500
# precision:  0.7164
# recall:  0.5518

echo "Evaluating precision recall: Training with 0.99 color and 0.01 gray; Sample gray"
python metrics/precision_recall.py --true ./datasets/cifar10/test_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.99_gray0.01/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.99_gray0.01_epoch_950_num10000_gray_images.npy \
--gpu 0 --batch-size 500
# precision: 0.7117
# recall: 0.583

echo "Evaluating precision recall: Training with 0.95 color and 0.05 gray; Sample color"
python metrics/precision_recall.py --true ./datasets/cifar10/test_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.95_gray0.05_epoch_950_num50000_color_images.npy \
--gpu 0 --batch-size 500
# precision: 0.6897
# recall: 0.6063

echo "Evaluating precision recall: Training with 0.95 color and 0.05 gray; Sample gray"
python metrics/precision_recall.py --true ./datasets/cifar10/test_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.95_gray0.05_epoch_950_num10000_gray_images.npy \
--gpu 0 --batch-size 500
# precision: 0.7002
# recall: 0.6448

echo "Evaluating precision recall: Training with 0.05 color and 0.95 gray; Sample gray"
python metrics/precision_recall.py --true ./datasets/cifar10/test_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.05_gray0.95_epoch_950_num50000_gray_images.npy \
--gpu 0 --batch-size 500
# precision: 0.755
# recall: 0.59

echo "Evaluating precision recall: Training with 0.05 color and 0.95 gray; Sample color"
python metrics/precision_recall.py --true ./datasets/cifar10/test_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.05_gray0.95_epoch_950_num10000_color_images.npy \
--gpu 0 --batch-size 500
# precision: 0.7191
# recall: 0.5333

echo "Evaluating precision recall: Training with 0.01 color and 0.99 gray; Sample gray"
python metrics/precision_recall.py --true ./datasets/cifar10/test_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.01_gray0.99/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.01_gray0.99_epoch_950_num50000_gray_images.npy \
--gpu 0 --batch-size 500
# precision: 0.7589
# recall: 0.5911

echo "Evaluating precision recall: Training with 0.01 color and 0.99 gray; Sample color"
python metrics/precision_recall.py --true ./datasets/cifar10/test_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.01_gray0.99/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.01_gray0.99_epoch_950_num10000_color_images.npy \
--gpu 0 --batch-size 500
# precision: 0.729
# recall: 0.4727

echo "Evaluating precision recall: Training with 0.0 color and 1.0 gray; Sample gray"
python metrics/precision_recall.py --true ./datasets/cifar10/test_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.0_gray1.0/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.0_gray1.0_epoch_950_num50000_images.npy \
--gpu 0 --batch-size 500
# precision: 0.738
# recall: 0.62


echo "Evaluating precision recall: Training with 1.0 color and 0.0 gray; Sample color"
python metrics/precision_recall.py --true ./datasets/cifar10/train_color.npy \
--fake ./logs/2022-12-08/cifar10/color1.0_gray0.0/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color1.0_gray0.0_epoch_950_num50000_images.npy \
--gpu 0 --batch-size 500
# precision:  0.738
# recall:  0.5492

echo "Evaluating precision recall: Training with 0.99 color and 0.01 gray; Sample color"
python metrics/precision_recall.py --true ./datasets/cifar10/train_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.99_gray0.01/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.99_gray0.01_epoch_950_num50000_color_images.npy \
--gpu 0 --batch-size 500
# precision:  0.7212
# recall:  0.5615

echo "Evaluating precision recall: Training with 0.99 color and 0.01 gray; Sample gray"
python metrics/precision_recall.py --true ./datasets/cifar10/train_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.99_gray0.01/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.99_gray0.01_epoch_950_num10000_gray_images.npy \
--gpu 0 --batch-size 500
# precision: 0.7292
# recall: 0.5762

echo "Evaluating precision recall: Training with 0.95 color and 0.05 gray; Sample color"
python metrics/precision_recall.py --true ./datasets/cifar10/train_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.95_gray0.05_epoch_950_num50000_color_images.npy \
--gpu 0 --batch-size 500
# precision: 0.6877
# recall: 0.6131

echo "Evaluating precision recall: Training with 0.95 color and 0.05 gray; Sample gray"
python metrics/precision_recall.py --true ./datasets/cifar10/train_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.95_gray0.05/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.95_gray0.05_epoch_950_num10000_gray_images.npy \
--gpu 0 --batch-size 500
# precision: 0.7105
# recall: 0.6331

echo "Evaluating precision recall: Training with 0.05 color and 0.95 gray; Sample gray"
python metrics/precision_recall.py --true ./datasets/cifar10/train_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.05_gray0.95_epoch_950_num50000_gray_images.npy \
--gpu 0 --batch-size 500
# precision: 0.7694
# recall: 0.5881

echo "Evaluating precision recall: Training with 0.05 color and 0.95 gray; Sample color"
python metrics/precision_recall.py --true ./datasets/cifar10/train_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.05_gray0.95/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.05_gray0.95_epoch_950_num10000_color_images.npy \
--gpu 0 --batch-size 500
# precision: 0.714
# recall: 0.5411

echo "Evaluating precision recall: Training with 0.01 color and 0.99 gray; Sample gray"
python metrics/precision_recall.py --true ./datasets/cifar10/train_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.01_gray0.99/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.01_gray0.99_epoch_950_num50000_gray_images.npy \
--gpu 0 --batch-size 500
# precision: 0.7723
# recall: 0.5915

echo "Evaluating precision recall: Training with 0.01 color and 0.99 gray; Sample color"
python metrics/precision_recall.py --true ./datasets/cifar10/train_color.npy \
--fake ./logs/2022-12-08/cifar10/color0.01_gray0.99/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.01_gray0.99_epoch_950_num10000_color_images.npy \
--gpu 0 --batch-size 500
# precision: 0.7277
# recall: 0.4846

echo "Evaluating precision recall: Training with 0.0 color and 1.0 gray; Sample gray"
python metrics/precision_recall.py --true ./datasets/cifar10/train_gray.npy \
--fake ./logs/2022-12-08/cifar10/color0.0_gray1.0/UNet_diffusionstep_1000_samplestep_250_condition_False_lr_0.0001_bs_512/samples/cifar10_color0.0_gray1.0_epoch_950_num50000_images.npy \
--gpu 0 --batch-size 500
# precision: 0.7473
# recall: 0.616