#_*_coding:utf-8_*_

#author: lgz

#date: 19-6-6


from __future__ import print_function, division
import torch
import numpy as np
import time
import os
import copy
from sklearn import metrics
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# def naive_auc(labels, preds):
#     """
# 　　　先排序，然后统计有多少正负样本对满足：正样本预测值>负样本预测值, 再除以总的正负样本对个数
#      复杂度 O(NlogN), N为样本数
#     """
#     n_pos = sum(labels)
#     n_neg = len(labels) - n_pos
#     total_pair = n_pos * n_neg
#
#     labels_preds = zip(labels, preds)
#     labels_preds = sorted(labels_preds, key=lambda x: x[1])
#     accumulated_neg = 0
#     satisfied_pair = 0
#     for i in range(len(labels_preds)):
#         if labels_preds[i][0] == 1:
#             satisfied_pair += accumulated_neg
#         else:
#             accumulated_neg += 1
#
#     return satisfied_pair / float(total_pair)
#
# def PR_curve(y,pred):
#     pos = np.sum(y == 1)
#     neg = np.sum(y == 0)
#     pred_sort = np.sort(pred)[::-1]  # 从大到小排序
#     index = np.argsort(pred)[::-1]  # 从大到小排序
#     y_sort = y[index]
#     # print(y_sort)
#
#     Pre = []
#     Rec = []
#     for i, item in enumerate(pred_sort):
#         if i == 0:#因为计算precision的时候分母要用到i，当i为0时会出错，所以单独列出
#             Pre.append(1)
#             Rec.append(0)
#
#         else:
#             Pre.append(np.sum((y_sort[:i] == 1)) /i)
#             Rec.append(np.sum((y_sort[:i] == 1)) / pos)
#
#     return Pre, Rec


def train_model(dataloaders, dataset_sizes, model, class_names, criterion, save_loss_path,
                            margin, optimizer, scheduler, num_epochs=25):

    #打开一个txt文件，存入训练过程相关值
    file_name = os.path.join(save_loss_path, 'loss.txt')
    # print(file_name)
    with open(file_name, 'w+') as file:
        file.write('====================LOSS=====================')

    #列表记录整个训练数据集所有样本的acc, auc等值，并用来作曲线走势图
    show_train_acc = []
    show_train_auc = []
    show_val_acc = []
    show_val_auc = []
    show_epoch = []

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_margin_wts = copy.deepcopy(margin.state_dict())

    best_acc = 0.0
    best_auc = 0.0

    for epoch in range(num_epochs):

        show_epoch.append(epoch)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        with open(file_name, 'a+') as file:
            file.write('\n')
            file.write('\n')
            file.write('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            with open(file_name, 'a+') as file:
                file.write('\n')
            print()

            if phase == 'train':
                scheduler.step()
                model.train()
                train_scores_auc = [] #记录所有训练样本的得分，用来计算最终AUC
                train_labels_auc = []
            else:
                model.eval()
                val_scores_auc = []
                val_labels_auc = []

            running_loss = 0.0
            running_corrects = 0

            class_correct = list(0.for i in range(len(class_names)))
            # print('class_correct', class_correct)
            class_total = list(0.for i in range(len(class_names)))

            # Iterate over data.
            n = 0
            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)
                n = list(labels.size())[0]

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    # margin is for chosing: Linear, arc_face_loss,etc
                    outputs = margin(outputs, labels)
                    loss = criterion(outputs, labels)
                    sout, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                c = (preds==labels)

                for i in range(n):

                    label = labels[i]
                    class_correct[label.item()] +=c[i].item()
                    class_total[label.item()] += 1

                ##########  for AUC ################
                #对每个批次的得分score和标签label进行存储到大list

                labels_cpu = (labels.data.cpu().numpy()).tolist()
                for each_label in labels_cpu:
                    if phase =='train':
                        train_labels_auc.append(each_label)
                    else :
                        val_labels_auc.append(each_label)


                softmax_score = outputs.detach()
                softmax_score = F.softmax(softmax_score, dim=1)

                #对于‘0’和‘1’两类，如果把‘1’当成正样本，
                indices = torch.tensor([1,]).to(device)
                sout2 = torch.index_select(softmax_score, dim = 1, index = indices)

                # # 对于‘0’和‘1’两类，如果把‘0’当成正样本，
                # indices = torch.tensor([0, ]).to(device)
                # sout2 = torch.index_select(softmax_score, dim=0, index=indices)

                scores_scale = sout2.detach()
                # scores_scale = (scores_scale - torch.min(scores_scale)) / (torch.max(scores_scale) - torch.min(scores_scale))
                scores_cpu = (scores_scale.data.cpu().numpy()).tolist()
                for each_score in scores_cpu:
                    if phase=='train':
                        train_scores_auc.append(each_score)
                    else :
                        val_scores_auc.append(each_score)

                #每个批次的得分score和标签label存储完毕
                ###########################################

            #获取整个数据集所有batch的得分score和对应的label
            if phase == 'train':
                scores_auc = train_scores_auc
                labels_auc = train_labels_auc
            else:
                scores_auc = val_scores_auc
                labels_auc = val_labels_auc


            scores_auc = np.array(scores_auc)
            # print("score: ", scores_auc)

            labels_auc = np.array(labels_auc)
            # print("labels: ", labels_auc)

            fpr, tpr, thresholds = metrics.roc_curve(labels_auc, scores_auc, pos_label= 1)  #pos_label= 0
            # print('fpr: ', fpr, '\n','tpr: ', tpr, '\n', 'th :', thresholds)
            # print(metrics.auc(fpr, tpr))

            AUC = metrics.auc(fpr, tpr)
            # AUC = naive_auc(labels_auc, scores_auc)

            # if AUC_ != AUC:
            #     print('SKlearn_AUC: ', AUC_)
            #     print('function_AUC: ', AUC)



            if phase=='val':

                plt.plot(fpr, tpr, marker='o')
                plt.savefig('train_results/ROC/'+'ROC({:.4f}) in epoch {:.0f} in validation.png'.format(AUC, epoch))
                plt.close()

                Pre, Rec, _ = metrics.precision_recall_curve(labels_auc, scores_auc)
                plt.plot(Rec, Pre, 'k')
                plt.plot([(0, 0), (1, 1)], 'r--')
                plt.xlim([-0.01, 1.01])
                plt.ylim([-0.01, 01.01])
                plt.ylabel('Precision')
                plt.xlabel('Recall')
                plt.savefig('train_results/PR/' + 'PR(AUC{:.4f}) in epoch {:.0f} in validation.png'.format(AUC, epoch))
                plt.close()

                # plt.show()

            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            #如果是两个分类，打印每个类别的acc
            for i in range(2):
                print('Accuracy of {} : {:.4f}  ({:.0f}/{:.0f})'
                      .format(class_names[i], 100 * class_correct[i] / class_total[i],class_correct[i], class_total[i]))


            #保存每个epoch的AUC，ACC，用来最后训练完画曲线走势图
            if phase == 'train':
                show_train_auc.append(AUC)
                show_train_acc.append(epoch_acc.item())
            else:
                show_val_acc.append(epoch_acc.item())
                show_val_auc.append(AUC)

            # 打印每个epoch的AUC，ACC
            print('{} Loss: {:.4f}   Acc: {:.4f}   AUC: {:.4f}'.format(phase, epoch_loss,
                                                                   epoch_acc, AUC ))

            # 将每个epoch的相关值写入train_results/loss.txt
            with open(file_name, 'a+') as loss_file:
                loss_file.write('\n')

                loss_file.write('{} Loss: {:.4f}   Acc: {:.4f}   AUC: {:.4f}'.format(phase, epoch_loss,
                                                                   epoch_acc, AUC))
                loss_file.write('\n')
                loss_file.write('Accuracy of {} : {:.4f}  ({:.0f}/{:.0f})'
                      .format(class_names[0], 100 * class_correct[0] / class_total[0],class_correct[0], class_total[0]))

                loss_file.write('\n')
                loss_file.write('Accuracy of {} : {:.4f}  ({:.0f}/{:.0f})'
                                .format(class_names[1], 100 * class_correct[1] / class_total[1], class_correct[1],
                                        class_total[1]))

                # loss_file.write('===================================================================\n')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                #保存最好acc的模型
                best_model_wts = copy.deepcopy(model.state_dict())
                best_margin_wts = copy.deepcopy(margin.state_dict())

            if phase == 'val' and AUC > best_auc:
                best_auc = AUC
                # 保存最好auc的模型
                # best_model_wts = copy.deepcopy(model.state_dict())
                # best_margin_wts = copy.deepcopy(margin.state_dict())

        print()

    plt.figure(figsize=(10, 5))
    plt.title("ACC and AUC During Training and validation")
    plt.plot(show_epoch,  show_train_acc, label="train_acccuracy")
    plt.plot(show_epoch, show_train_auc, label="train_AUC")
    plt.plot(show_epoch, show_val_acc, label="val_acccuracy")
    plt.plot(show_epoch, show_val_auc, label="val_AUC")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend(loc='lower right', fontsize=16)
    plt.savefig('train_results/ACC and AUC.png')
    plt.show()


    time_elapsed = time.time() - since
    print('Cost time: {:.0f}hours ({:.0f}minutes {:.0f}s)'.format(time_elapsed // 3600, time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val AUC: {:4f}'.format(best_auc))

    with open(file_name, 'a+') as loss_file:
        loss_file.write('\n')
        loss_file.write('\n')
        loss_file.write('Cost time {:.0f}hours ({:.0f}minutes {:.0f}s)'.format(time_elapsed // 3600, time_elapsed // 60, time_elapsed % 60))
        loss_file.write('\n')
        loss_file.write('Best val Acc: {:4f}'.format(best_acc))
        loss_file.write('\n')
        loss_file.write('Best val AUC: {:4f}'.format(best_auc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    margin.load_state_dict(best_margin_wts)


    return model, margin