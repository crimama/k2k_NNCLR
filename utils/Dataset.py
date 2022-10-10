import os 
import warnings 
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd 
import cv2 

import matplotlib.pyplot as plt 

from PIL import Image

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
import torchvision 
from torchvision import transforms
from torchvision.datasets import STL10
import timm

class Custom_Dset(Dataset):
    def __init__(self,dataset):
        super().__init__
        self.dataset = dataset 
        self.augmenter = self.__augmenter__() 

    def __len__(self):
        return len(self.dataset)

    def __augmenter__(self):
        augmentation = transforms.Compose([
            transforms.ToTensor()
        ])
        return augmentation

    def __getitem__(self,idx):
        img,label = self.dataset[idx]
        img = self.augmenter(img)

        return img, label 
        
def prepare_dataloader(batch_size):
    root = './Data'
    #train - label 
    label_train_stl10 = STL10(root=root,split='train')
    label_train_Dset = Custom_Dset(label_train_stl10)
    label_train_loader = DataLoader(label_train_Dset,batch_size=batch_size,shuffle=True)

    #train-unlabelled 
    unlabel_train_stl10 = STL10(root=root,split='unlabeled')
    unlabel_train_Dset = Custom_Dset(unlabel_train_stl10)
    unlabel_train_loader = DataLoader(unlabel_train_Dset,batch_size=batch_size,shuffle=True)

    #test 
    test_st10 = STL10(root=root,split='test')
    test_dset = Custom_Dset(test_st10)
    test_loader = DataLoader(test_dset,batch_size=batch_size,shuffle=False)

    print(len(label_train_stl10), len(unlabel_train_stl10),len(test_st10))
    
    return label_train_loader, unlabel_train_loader , test_loader