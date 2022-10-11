import os 
import warnings 
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd 
import cv2 
from tqdm import tqdm 
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

#데이터 로더 
from utils import prepare_dataloader

#어그먼테이션 
from utils import augmenter

#model 
from utils import ResnetEncoder,ProjectionHead,PredictionHead,NNCLR

#loss, nn 
from utils import nearest_neighbour,contrastive_loss 
from utils import NTXentLoss

from pytorch_lars import LARS 
from info_nce import InfoNCE

#memorybank 
from utils import NNMemoryBankModule


# 하이퍼 파라미터 

shuffle_buffer = 5000 
labelelled_train_images = 5000 
unlabelled_images = 100000

temperature = 0.1 
queue_size = 10000
contrastive_augmenter = {
    "brightness" : 0.5, 
    "name" : "contrastive_augmenter",
    "scale" : (0.2,1.0)
}
classification_augmenter = {
    "brightness": 0.2,
    "name": "classification_augmenter",
    "scale": (0.5, 1.0),
}
input_shape = (96,96,3)
width = 128 
num_epochs = 500 
steps_per_epoch = 200 
batch_size= 128
learning_rate = 1e-3 
device = 'cuda:0'
feature_dimensions = 256 


def model_save(model,model_name):
    torch.save(model,f'save_models/{model_name}')

if __name__ == "__main__":
    #데이터 로더 로드 
    label_train, unlabel_train, test = prepare_dataloader(batch_size=batch_size)

    #Augmentation 로드 
    contrastive_augmentation = augmenter(contrastive_augmenter)
    classification_augmentation = augmenter(classification_augmenter)

    #모델 로드 
    model = NNCLR(device).to(device)
    memory_bank = NNMemoryBankModule().to(device)

    #compile 
    criterion = InfoNCE()
    optimizer = LARS(model.parameters(),lr=learning_rate,momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=0)

    #Train 
    for epoch in tqdm(range(num_epochs)):
        model.train() 
        epoch_loss = 0.
        n = 0 
        #이미지 데이터 로드 
        for (label_img,labels),(unlabel_img,_) in zip(label_train,unlabel_train):
            label_img,unlabel_img = label_img.to(device),unlabel_img.to(device)

            images = torch.concat((label_img,unlabel_img),axis=0).to(device)
            augmented_images_1 = contrastive_augmentation(images).to(device)
            augmented_images_2 = contrastive_augmentation(images).to(device)
        

            optimizer.zero_grad()

            z0,p0 = model(augmented_images_1)
            z1,p1 = model(augmented_images_2)

            z0 = memory_bank(z0,update=False)
            z1 = memory_bank(z1,update=True)
            loss = 0.5 * (criterion(z0,p1) + criterion(z1,p0))
            

            loss.backward()
            optimizer.step() 
            epoch_loss += loss 
            n+=1 
        print(f'Epoch : {epoch} | loss : {epoch_loss/n}')
        if epoch == 0:
            best = epoch_loss/n
            model_save(model,model_name='best.pt')
        if epoch_loss/n < best:
            model_save(model,model_name='best.pt')
            print(f'model save at {epoch}')
        
    
