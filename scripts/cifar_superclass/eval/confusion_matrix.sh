for lr in 0.01 0.005 0.001; do 
    for wd in 0.001 0.0001 0.00001; do 
        CUDA_VISIBLE_DEVICES=0 python -m visualize.confusion_matrix \
        --num-classes 10 --mode confusion-matrix --num-gpus 1 --date 2023-04-20 \
        --num-epochs 20 --batch-size 128 --learning-rate $lr --weight-decay $wd \
        --ckpt-path ./logs/2023-04-20/cifar-superclass/simple_classifier/bs128_lr${lr}_decay${wd}/ckpt/model_param_final.pth
    done;
done