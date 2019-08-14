import os
import cv2
import torch
import random
import imutils
import torchvision

import numpy as np
import matplotlib.pyplot as plt

from skimage import img_as_float

class Resize(object):
    def __init__(self, size):
        assert len(size) == 2
        self.size=size
    
    def __call__(self, img):
        return cv2.resize(img, self.size[::-1])
    
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            return cv2.flip(img,1)#img[:,::-1]
        return img

class ToTensor(object):
    def __init__(self):
        pass
        
    def __call__(self, img):
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        return img.float()
    
class Grayscale(object):
    def __init__(self):
        pass
        
    def __call__(self, img):
        img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(float)
        return np.expand_dims(img,2)

def adjust_brightness_and_contrast(img, brightness, contrast):
    img = img_as_float(img) - 0.5
    return np.clip(0.5 + img * (contrast/2+1)  + brightness,0,1)
        
class ColorJitter(object):
    def __init__(self, brightness, contrast):
        self.brightness = brightness
        self.contrast = contrast        
        
    def __call__(self, img):
        brightness = random.uniform(-self.brightness,self.brightness)
        contrast = random.uniform(-self.contrast,self.contrast)
        return adjust_brightness_and_contrast(img, brightness, contrast)
        

class RandomRotationandScaling(object):
    def __init__(self, rotation, scaling, background='white'):
        assert background in ['white','neutral','black']
        self.rotation=rotation
        self.scaling=scaling
        if background == 'white':
            self.background_rgb = [1,1,1]
        elif background == 'neutral':
            self.background_rgb = [138.14/255, 120.80/255, 103.32/255]
        elif background == 'black':
            self.background_rgb = [0,0,0]
        
        
    def __call__(self, img):
        rotation = random.uniform(-self.rotation, self.rotation)
        scale = random.uniform(1-self.scaling,1+self.scaling)
        
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, scale)
        
        n_channels = img.shape[2]
        assert n_channels in [3,6]
        if n_channels == 6:
            img1 = cv2.warpAffine(img[:,:,:3], M, (cols, rows), borderValue=self.background_rgb)
            img2 = cv2.warpAffine(img[:,:,3:], M, (cols, rows), borderValue=self.background_rgb)
            return  np.concatenate([img1,img2],axis=2)
        elif n_channels == 3:
            img1 = cv2.warpAffine(img[:,:,:3], M, (cols, rows), borderValue=self.background_rgb)
            return  img1
        
        
class GaussianNoise(object):
    def __init__(self):
        pass
        
        
    def __call__(self):
        pass
        
        
class MaskingNoise(object):
    def __init__(self):
        pass
        
        
    def __call__(self):
        pass