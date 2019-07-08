#_*_coding:utf-8_*_

#author: lgz

#date: 19-6-6

from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import os
from trainer import train_model
from visualization import visualize_model

from losses.cos_face_loss import  CosineMarginProduct
from losses.arc_face_loss import  ArcMarginProduct
from losses.linear_loss import InnerProduct

from config import TrainOptions
args = TrainOptions().parse()

from datasets import CreateDataloader
dataloaders, dataset_sizes, class_names = CreateDataloader(args)

print('class_names: ', class_names)

save_model = args.save_model
if not os.path.exists(save_model):
    os.mkdir(save_model)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(168)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##############################################################################################################
#注意到已经把backbone最后一层的linear注释掉，而使用margin层（可看成带要学习参数的特殊linear层）代替
#比如用于1000分类的resnet18最后使用linear(x, 1000), 把这层拿掉， 这里的x就是模型的输出维度，也就是下面的model_output_dimension

if args.backbone == 'resnet18':
    from backbones.resnet import resnet18
    model_ft = resnet18()
    pretrained_dict = models.resnet18(pretrained=True).state_dict()
    model_output_dimension = 512

elif args.backbone == 'densenet169':
    from backbones.densenet import densenet169
    model_ft = densenet169()
    pretrained_dict = models.densenet169(pretrained=True).state_dict()
    model_output_dimension = 1664

elif args.backbone == 'inception_v4':
    from backbones import inception_v4
    model_ft = inception_v4.inception_v4(num_classes = args.numclass, pretrained = False)
    pretrained_dict = inception_v4.inception_v4(num_classes = args.numclass).state_dict()
    model_output_dimension = 1536

elif args.backbone == 'senet154':
    from backbones import senet
    model_ft = senet.senet154(num_classes = args.numclass, pretrained = None)
    pretrained_dict = senet.senet154().state_dict()
    model_output_dimension = 2048

elif args.backbone == 'nasnet':
    from backbones import nasnet
    model_ft = nasnet.nasnetalarge(num_classes = args.numclass, pretrained = False)
    pretrained_dict = nasnet.nasnetalarge().state_dict()
    model_output_dimension = 4032

else:
    print(args.backbone, ' is not available!')

# print(model_ft)
###########################################

model_dict = model_ft.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model_ft.load_state_dict(model_dict)

total = sum([param.nelement() for param in model_ft.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))

############################################

if args.finetune_last_layer == True:

    for param in model_ft.parameters():
        param.requires_grad = False

##########################################
#严格来说，这里不是选择损失函数类型，而是选择最后一层是不是linear
#也许是不同的linear带来不同的损失，人们常常叫margin loss

if args.loss_type == 'ArcFace':
    margin = ArcMarginProduct(model_output_dimension, args.numclass)
elif args.loss_type == 'CosFace':
    margin = CosineMarginProduct(model_output_dimension, args.numclass)
elif args.loss_type == 'SphereFace':
    pass
elif args.loss_type == 'Softmax':
    margin = InnerProduct(model_output_dimension, args.numclass)
else:
    print(args.loss_type, 'is not available!')

#################################################

model_ft = model_ft.to(device)
margin = margin.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD([
        {'params': model_ft.parameters(), 'weight_decay': 0.000001},
        {'params': margin.parameters(), 'weight_decay': 0.000001}
], lr=0.001, momentum=0.9, nesterov=True)


# Decay LR by a factor of 0.1 every 24 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=24, gamma=0.1)


model_ft, margin = train_model(dataloaders, dataset_sizes,  model_ft, class_names, criterion,
                               args.save_train_results_dir, margin,
                               optimizer_ft, exp_lr_scheduler, num_epochs=args.num_epoch)

torch.save(model_ft.state_dict(), save_model+'/' + 'finetune_model.pkl')
torch.save(margin.state_dict(), save_model + '/' + 'finetune_model_margin.pkl')

#训练完可以看一些验证结果的可视化预测情况
visualize_model(model_ft, dataloaders, class_names, margin)

