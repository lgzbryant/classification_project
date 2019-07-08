#!/bin/bash
################################ training ################################
# use command:  ./train.sh 

python train.py 
--backbone 'resnet18'
--resize 224
--dataset_dir data/cifar10
--loss_type Softmax
