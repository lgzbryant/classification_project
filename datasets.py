#_*_coding:utf-8_*_

#author: lgz

#date: 19-6-10

from torch.utils.data import Dataset
import torch
import torchvision
from torchvision import datasets, models, transforms
import os


def CreateTestDataloader(args):

    data_transforms = {

        'test': transforms.Compose([
            transforms.Resize(args.resize),
            # transforms.CenterCrop(args.resize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0.5, 0.5, 0.5],
            #                      [0.5, 0.5, 0.5])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(args.dataset_dir, x),
                                              data_transforms[x]) for x in ['test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batchsize, shuffle=True, num_workers=4)
                   for x in ['test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}

    # print(dataset_sizes)

    print("Total number of test images in the dataset:", dataset_sizes['test'])

    class_names = image_datasets['test'].classes

    return dataloaders, dataset_sizes, class_names


def CreateDataloader(args):

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(args.resize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0.5, 0.5, 0.5],
            #                      [0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            transforms.Resize(args.resize),
            # transforms.CenterCrop(args.resize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0.5, 0.5, 0.5],
            #                      [0.5, 0.5, 0.5])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(args.dataset_dir, x),
                                              data_transforms[x]) for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batchsize, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # print(dataset_sizes)
    print("Total number of train images in the dataset:", dataset_sizes['train'])
    print("Total number of val images in the dataset:", dataset_sizes['val'])

    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names

