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



def nearest_neighbour(projections,feature_queue):
    support_similarities = torch.matmul(projections,feature_queue.T)
    nn_projections = torch.gather(feature_queue,torch.argmax(support_similarities,axis=1),axis=0)
    
    return_value = projections + (nn_projections - projections).detach()

    return return_value




def contrastive_loss(projection_1,projection_2,temperature,feature_queue):
    
    projection_1 = F.normalize(projection_1,p=2)
    projection_2 = F.normalize(projection_2,p=2)

    similarities_1_2_1 = (torch.matmul(nearest_neighbour(projection_1,feature_queue),projection_2)/temperature)
    similarities_1_2_2 = (torch.matmul(projection_2,nearest_neighbour(projection_1,feature_queue))/temperature)
    similarities_2_1_1 = (torch.matmul(nearest_neighbour(projection_2,feature_queue),projection_1)/temperature)
    similarities_2_1_2 = (torch.matmul(projection_1,self.nearest_neighbour(projection_2,feature_queue))/temperature)

    contrastive_batch_size= projection_1.shape[0]
    contrastive_labels = torch.range(0,contrastive_batch_size)
    loss = nn.CrossEntropyLoss(
        torch.concat(
            [
            contrastive_labels,
            contrastive_labels,
            contrastive_labels,
            contrastive_labels,
            ],
            axis=0),
        torch.concat(
            [
            similarities_1_2_1,
            similarities_1_2_2,
            similarities_2_1_1,
            similarities_2_1_2,
            ],
            axis=0
        ),
    )

    feature_quene.assign(torch.concat([projection_1,feature_quene[:-batch_size]],axis=0))
    return loss,feature_queue

    