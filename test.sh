#!/bin/bash
################################ testing ################################
# use command:  ./test.sh 

python test.py --backbone resnet18 --loss_type Softmax --dataset_dir data/cifar10 --test_model save_model/finetune_model.pkl --margin_model save_model/finetune_model_margin.pkl --resize 224 --numclass 2 --batchsize 24

