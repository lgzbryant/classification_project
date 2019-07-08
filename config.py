#_*_coding:utf-8_*_

#author: lgz

#date: 19-6-10


import argparse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):

        self.initialized = True

    def parse(self):

        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        args = vars(self.opt)

        print('======================Check options========================')
        print('')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('')
        print('======================check done============================')


        return self.opt

class TrainOptions(Options):
    def initialize(self):
        Options.initialize(self)

        self.parser.add_argument('--numclass', action="store", type=int,
                                default= 2,
                                help="the number of class, for anomaly detection is 2 ")

        self.parser.add_argument('--backbone', type=str, default='resnet18',
                                 help='resnet18, densenet169, inception_v4, nasnet, senet154')

        self.parser.add_argument("--num_epoch", action="store", type=int,
                                 default= 2 ,
                                 help="training epoches")

        self.parser.add_argument("--finetune_last_layer", action="store", type=bool,
                                 default=False,
                                 help="if True, just finetune the last layers")

        self.parser.add_argument("--resize", action="store", type=int,
                                default= 224,
                                help="resize input")

        self.parser.add_argument("--dataset_dir", action="store", type=str,

                                default="data/cifar10",

                                help="cifar10, membrane, hymenoptera_data, where the dataset put in ")

        self.parser.add_argument("--save_model", action="store", type=str,
                                default="save_model",
                                help="where the model save ")

        self.parser.add_argument("--batchsize", action="store", type=int,
                                 default=24,
                                 help="training batchsize")

        self.parser.add_argument('--loss_type', type=str, default='Softmax',
                                 help='ArcFace, CosFace, Softmax')

        self.parser.add_argument('--save_train_results_dir', type=str, default='train_results',
                                 help='record the train value ')


class TestOptions(Options):
    def initialize(self):
        Options.initialize(self)

        self.parser.add_argument('--numclass', action="store", type=int,
                                default= 2,
                                help="the number of class, for anomaly detection is 2 ")

        self.parser.add_argument('--backbone', type=str, default='resnet18',
                                 help='resnet18, densenet169, inception_v4, nasnet, senet154')


        self.parser.add_argument("--resize", action="store", type=int,
                                default= 224,
                                help="resize input")

        self.parser.add_argument("--dataset_dir", action="store", type=str,

                                default="data/cifar10",

                                help="cifar10, membrane, hymenoptera_data, where the dataset put in ")

        self.parser.add_argument("--test_model", action="store", type=str,
                                default='save_model/finetune_model.pkl',
                                help="where the model save ")

        self.parser.add_argument("--margin_model", action="store", type=str,
                                 default='save_model/finetune_model_margin.pkl',
                                 help="where the margin model save ")


        self.parser.add_argument("--batchsize", action="store", type=int,
                                 default=24,
                                 help="training batchsize")

        self.parser.add_argument('--loss_type', type=str, default='Softmax',
                                 help='ArcFace, CosFace, Softmax')








