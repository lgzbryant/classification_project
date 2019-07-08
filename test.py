#_*_coding:utf-8_*_

#author: lgz

#date: 19-6-6

from __future__ import print_function, division

import torch
import numpy as np
import random
from tester import test_model
from losses.cos_face_loss import  CosineMarginProduct
from losses.arc_face_loss import  ArcMarginProduct
from losses.linear_loss import InnerProduct

from config import TestOptions
args = TestOptions().parse()

from datasets import CreateTestDataloader
dataloaders, dataset_sizes, class_names = CreateTestDataloader(args)

print('class_names: ', class_names)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(168)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##########################################

from backbones.resnet import resnet18
model_ft = resnet18()

pretrained_dict = torch.load(args.test_model)
model_dict = model_ft.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model_ft.load_state_dict(model_dict)
print()
print('load model......')
print('load the model successfully!!!')
total = sum([param.nelement() for param in model_ft.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))
print()
# print(model_ft)
###########################################


##########################################
flag = True
if args.loss_type == 'ArcFace':
    margin = ArcMarginProduct(512, args.numclass)

elif args.loss_type == 'CosFace':
    margin = CosineMarginProduct(512, args.numclass)
elif args.loss_type == 'SphereFace':
    pass
elif args.loss_type == 'Softmax':
    margin = InnerProduct(512, args.numclass)
else:
    flag = False
    print(args.loss_type, 'is not available!')

if flag:
    pretrained_margin_dict = torch.load(args.margin_model)
    margin_dict = margin.state_dict()
    margin_dict.update(pretrained_margin_dict)
    margin.load_state_dict(margin_dict)
    print('load margin......')
    print('load the margin successfully!!!')
    total = sum([param.nelement() for param in margin.parameters()])
    print("Number of parameter: %.2f" % (total))
    print()

#################################################

model_ft = model_ft.to(device)
margin = margin.to(device)
test_model(dataloaders, dataset_sizes,  model_ft, class_names,  margin)




