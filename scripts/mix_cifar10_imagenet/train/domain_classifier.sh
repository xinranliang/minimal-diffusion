CUDA_VISIBLE_DEVICES=0 python dataset/domain_classifier.py \
--arch resnet50 --dataset cifar10-imagenet --num-domains 2 --mode train \
--num-epochs 10 --batch-size 2048 --learning-rate 1e-3 --weight-decay 1e-4 \
--num-gpus 1 --date 2023-04-08