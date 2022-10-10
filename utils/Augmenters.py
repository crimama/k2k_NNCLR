import torch 
from torchvision import transforms
import torchvision 


def augmenter(kwargs):
    brightness,_,scale = kwargs.values()
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=96,scale=scale),
        transforms.ColorJitter(brightness=brightness)
    ])
    