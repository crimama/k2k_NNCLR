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





class ResnetEncoder(nn.Module):
    def __init__(self,vector_size=2048):
        super(ResnetEncoder,self).__init__()
        self.resnet50 = res50 = timm.create_model('resnet50',num_classes=vector_size,pretrained=True)

    def forward(self,x):
        x = self.resnet50(x)
        return x     

class ProjectionHead(nn.Module):
    def __init__(self):
        super(ProjectionHead,self).__init__()
        self.fc1 = self.linear_layer()
        self.fc2 = self.linear_layer()
        self.fc3 = nn.Sequential(nn.Linear(in_features=2048,out_features=256),
                                 nn.BatchNorm1d(256))

    def linear_layer(self,out_features=2048):
        Linear_layer = nn.Sequential(
            nn.Linear(in_features=2048,out_features=out_features),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )
        return Linear_layer 

    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x 

class PredictionHead(nn.Module):
    def __init__(self):
        super(PredictionHead,self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=2048,out_features=4096),
            nn.BatchNorm1d(4096),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(in_features=4096,out_features=256)
    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x 

class NNCLR(nn.Module):
    def __init__(self,device='cpu'):
        super(NNCLR,self).__init__()
        self.encoder = ResnetEncoder()
        self.projection_head = ProjectionHead().to(device)
        self.prediction_head = PredictionHead().to(device)

    def forward(self,x):
        x = self.encoder(x)
        z = self.projection_head(x)
        p = self.prediction_head(x)
        return z,p