import argparse
import os


class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--data_root', type=str, default=r"image_all",
                                 help='dir of the dataset')
        self.parser.add_argument('--exp_name', type=str, default="image_data_01_27", help='the name of the experiment')
        self.parser.add_argument('--img_size', type=int, default='256', help=' where to save the result images')
        self.parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='size of the batches')
        self.parser.add_argument('--epochs', type=int, default=300, help='size of the batches')
        self.parser.add_argument('--lr_decay_epoch', type=int, default=100, help='epoch after which lr is decayed')
        self.parser.add_argument('--weight_decay', type=float, default=1e-5, help='L2 regularization')
        self.parser.add_argument('--save_interval', type=int, default=10, help='interval to save checkpoints')
        self.parser.add_argument('--pretrain_path', type=str, default=None, help='path to pretrained weights')

    def parse(self):
        if not self.initialized:
            self.initialize()
        args = self.parser.parse_args()
        self.args = args
        return self.args