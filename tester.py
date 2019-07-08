#_*_coding:utf-8_*_

#author: lgz

#date: 19-6-20


from __future__ import print_function, division
import torch
import numpy as np
import time
import os
from torchvision import transforms
from sklearn import metrics
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from PIL import Image
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tensor2im(image_tensor):

    inp = image_tensor.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    return inp

def save_image(image_numpy, image_path, pred, label):
    plt.title('label: {} predict: {}'.format(label, pred))
    plt.imshow(image_numpy)
    plt.savefig(image_path)
    plt.close()


def test_model(dataloaders, dataset_sizes, model, class_names, margin):

    since = time.time()

    print('predicting......')

    model.eval()   # Set model to evaluate mode
    running_corrects = 0

    show_val_acc = []
    show_val_auc = []

    class_correct = list(0.for i in range(len(class_names)))
    # print('class_correct', class_correct)
    class_total = list(0.for i in range(len(class_names)))

    n = 0
    test_labels_auc = []
    test_scores_auc = []

    predict_wrong_image_number = 0
    predict_wrong_image_count = 0

    n_batch = 1

    for inputs, labels in dataloaders['test']:

        print('deal with the {} batch ({} images)'.format(n_batch, len(labels)))
        n_batch += 1

        inputs = inputs.to(device)
        labels = labels.to(device)
        n = list(labels.size())[0]

        with torch.set_grad_enabled(False):

            outputs = model(inputs)

            # margin is for chosing: Linear, arc_face_loss,etc
            outputs = margin(outputs, labels)
            sout, preds = torch.max(outputs, 1)

            for i in range(len(labels)):
                title = predict_wrong_image_number * n + i

                if labels[i] != preds[i] :
                    predict_wrong_image_count += 1
                    image_numpy = tensor2im(inputs[i])
                    save_image(image_numpy, 'test_results/predict_wrong_images_results/' + str(title) + '.png',
                               preds[i].data.cpu().numpy(), labels[i].data.cpu().numpy())

            predict_wrong_image_number += 1

        running_corrects += torch.sum(preds == labels.data)

        c = (preds==labels)
        for i in range(n):
            label = labels[i]
            # print('!!!! ', label)
            class_correct[label.item()] +=c[i].item()
            class_total[label.item()] +=1


        #对每个批次的得分score和标签label进行存储到大list

        labels_cpu = (labels.data.cpu().numpy()).tolist()
        for each_label in labels_cpu:
                test_labels_auc.append(each_label)

        ##########  for AUC ##############
        softmax_score = outputs.detach()
        softmax_score = F.softmax(softmax_score, dim=1)
        # print(softmax_score)
        indices = torch.tensor([1,]).to(device)
        sout2 = torch.index_select(softmax_score, dim = 1, index = indices)
        # print(sout2)
        ###################################


        scores_scale = sout2.detach()
        # scores_scale = (scores_scale - torch.min(scores_scale)) / (torch.max(scores_scale) - torch.min(scores_scale))
        scores_cpu = (scores_scale.data.cpu().numpy()).tolist()
        for each_score in scores_cpu:
                test_scores_auc.append(each_score)

    #获取整个数据集所有batch的得分score和对应的label

    scores_auc = np.array(test_scores_auc)
    # print("score: ", scores_auc)
    labels_auc = np.array(test_labels_auc)
    # print("labels: ", labels_auc)

    fpr, tpr, thresholds = metrics.roc_curve(labels_auc, scores_auc, pos_label= 1)  #pos_label= 0
    # print('fpr: ', fpr, '\n','tpr: ', tpr, '\n', 'th :', thresholds)
    # print(metrics.auc(fpr, tpr))
    AUC = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, marker='o')
    plt.savefig('test_results/'+'ROC({:.4f}) in validation.png'.format(AUC))
    plt.close()

    Pre, Rec, _ = metrics.precision_recall_curve(labels_auc, scores_auc)
    plt.plot(Rec, Pre, 'k')
    plt.plot([(0, 0), (1, 1)], 'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 01.01])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig('test_results/' + 'PR(AUC{:.4f}) in validation.png'.format(AUC))
    plt.close()

    epoch_acc = running_corrects.double() / dataset_sizes['test']

    for i in range(2):
        print('Accuracy of {} : {:.4f}  ({:.0f}/{:.0f})'
              .format(class_names[i], 100 * class_correct[i] / class_total[i],class_correct[i], class_total[i]))

    show_val_acc.append(epoch_acc.item())
    show_val_auc.append(AUC)

    print('{}  Acc: {:.4f}   AUC: {:.4f}'.format('test', epoch_acc, AUC ))
    print('{} image(s) is/are mispredicted !!!!!!'.format(predict_wrong_image_count))
    print()

    time_elapsed = time.time() - since
    print('Cost time: {:.0f}hours ({:.0f}minutes {:.0f}s)'.format(time_elapsed // 3600, time_elapsed // 60, time_elapsed % 60))
