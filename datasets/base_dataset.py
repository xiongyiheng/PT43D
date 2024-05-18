
import numpy as np
from imageio import imread
from PIL import Image

from termcolor import colored, cprint

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

from torchvision import datasets

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def CreateDataset(opt):
    dataset = None

    # decide resolution later at model
        
    if opt.dataset_mode == 'snet_img':
        from datasets.snet_dataset import ShapeNetImgDataset
        train_dataset = ShapeNetImgDataset()
        test_dataset = ShapeNetImgDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat)
        test_dataset.initialize(opt, 'test', cat=opt.cat)

    elif opt.dataset_mode == 'snet':
        from datasets.snet_dataset import ShapeNetDataset
        train_dataset = ShapeNetDataset()
        test_dataset = ShapeNetDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat)
        test_dataset.initialize(opt, 'test', cat=opt.cat)

    elif opt.dataset_mode == 'scannet_img':
        from datasets.scannet import ScanNetImgDataset
        train_dataset = ScanNetImgDataset()
        test_dataset = ScanNetImgDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat)
        test_dataset.initialize(opt, 'test', cat=opt.cat)

    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    cprint("[*] Dataset has been created: %s" % (train_dataset.name()), 'blue')
    return train_dataset, test_dataset
