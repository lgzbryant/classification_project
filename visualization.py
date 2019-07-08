#_*_coding:utf-8_*_

#author: lgz

#date: 19-6-6

from __future__ import print_function, division
import torch
import numpy as np
import matplotlib.pyplot as plt

import cv2
from torch.autograd import Variable
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(2)  # pause a bit so that plots are updated


def visualize_model(model, dataloaders, class_names,  margin = None, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            if margin!=None:
                outputs = margin(outputs, labels)

            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])


                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)



def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


class FeatureVisualization():
    def __init__(self, img_path, selected_layer):
        self.img_path = img_path
        self.selected_layer = selected_layer
        # self.pretrained_model = models.resnet18(pretrained=True)
        ############################

        from backbones.resnet import resnet18
        model_ft = resnet18()

        pretrained_dict = torch.load('save_model/finetune_model.pkl')
        model_dict = model_ft.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model_ft.load_state_dict(model_dict)

        #############################

        self.pretrained_model = model_ft

        # print(self.pretrained_model)

    def process_image(self):
        img = cv2.imread(self.img_path)
        img = preprocess_image(img)
        return img

    def get_feature(self):
        # input = Variable(torch.randn(1, 3, 224, 224))
        input = self.process_image()
        print(input.shape)
        x = input

        # for index, layer in enumerate(self.pretrained_model.features):
        #     print("============ ", layer)
        #     x = layer(x)
        #     if (index == self.selected_layer):
        #         return x

        for name, module in self.pretrained_model._modules.items():
            print('name: ', name)
            x = module(x)
            if (name == self.selected_layer):
                return x

    def save_feature_to_image(self):
        features = self.get_feature()
        print("The selected layer output shape is : ", features.shape)

        for i in range(features.shape[1]):
            feature = features[:, i, :, :]

            # feature = features[:, 0, :, :]
            # print("select one channel: ", feature.shape)

            feature = feature.view(feature.shape[1], feature.shape[2])
            # print("view the feature map: ", feature.shape)

            # to numpy
            feature = feature.data.numpy()

            # use sigmod to [0,1]
            feature = 1.0 / (1 + np.exp(-1 * feature))

            # to [0,255]
            feature = np.round(feature * 255)
            # print(feature[0])
            # print("write the feature as image: ", feature)

            cv2.imwrite('./visual_feature_map/channel_' + str(i) + '.jpg', feature)



if __name__ == '__main__':
    # get class
    myClass = FeatureVisualization('healthy.bmp', 'layer1')
    print(myClass.pretrained_model)

    myClass.save_feature_to_image()