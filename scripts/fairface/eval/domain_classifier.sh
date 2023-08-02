
for lr in 0.01 0.005 0.001 0.0005 0.0001; do 
    for wd in 0.001 0.0001 0.00001; do 
        CUDA_VISIBLE_DEVICES=0 python -m domain_classifier.fairface \
            --arch resnet18 --pretrained --dataset fairface --num-domains 2 --mode test \
            --num-epochs 20 --batch-size 64 --learning-rate $lr --weight-decay $wd \
            --num-gpus 1 --date 2023-07-20
    done;
done;