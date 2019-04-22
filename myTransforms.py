import os
import cv2
import torch
import torch.optim as optim
import torchvision.transforms
import torch.nn.functional as F

from glob import glob
from torch.nn.functional import binary_cross_entropy

class Resize(object):
    def __init__(self, size):
        assert len(size) == 2
        self.size=size
    
    def __call__(self, img):
        print(type(img))
        print(img)
        return cv2.resize(img, self.size)
    
    
    