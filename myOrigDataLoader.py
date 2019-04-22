#Adapted from: https://github.com/githubharald/SimpleHTR/blob/master/src/DataLoader.py
import os
import cv2
import math
import random
import numpy as np
import pandas as pd

from glob import glob
from pathlib import Path

cwd = os.getcwd()

class Sample:
    "sample from the dataset"
    def __init__(self, imgPath, data):
        self.imgPath = imgPath
        self.data = data


class Batch:
    "batch of images and ground truth saliency"
    def __init__(self, imgPaths, imgs, data, batchRange):
        self.imgPaths = imgPaths
        self.imgs = np.stack(imgs, axis=0)
        self.data = data
        self.range = batchRange

class myDataLoader:
    "loads data "
    def __init__(self, batchSize, imgSize, trainSplit=0.8, SEED = 42, sides = 'both', datasets = ['international','copenhagen','aarhus']):
        assert sides in ['both','D','V']
        print('Starting DataLoader...')
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.trainSplit = trainSplit
        
        self.currIdx = 0
        self.imgPaths = []
        self.samples = []
        self.trainSamples = []
        self.valSamples = []
        self.saved = []
        
        for dataset in datasets:
            if sides == 'both':
                self.imgPaths += glob(f'/home/rob/thesis/datasets/{dataset}/4_color_corrected/*.jpg')
            else:
                self.imgPaths += glob(f'/home/rob/thesis/datasets/{dataset}/4_color_corrected/*_{sides}.jpg')
            
        self.dataFilePath = '/home/rob/Dropbox/thesis/2. code/src/data/data.xlsx'
        
        self.df = self.getData(self.dataFilePath)
              
        for imgPath in self.imgPaths:
            _id = os.path.basename(imgPath).split('_')[0]
            dataRow = self.df[self.df['ID (number)'].map(str) == _id].iloc[0]
            self.samples += [Sample(imgPath, dataRow)]
            
        random.seed(SEED)
        random.shuffle(self.samples)
        splitIdx = int(self.trainSplit*len(self.samples))
        self.trainSamples = self.samples[:splitIdx]
        self.valSamples = self.samples[splitIdx:]
         
        print("%s images found"%len(self.samples))
        print("Split into %s train and %s validation images with seed %s"%(len(self.trainSamples), len(self.valSamples), SEED))
        
        self.trainSet()
        
    def trainSet(self):
        self.currIdx = 0
        self.samples = self.trainSamples
        random.shuffle(self.samples)
        
    def validationSet(self):
        self.currIdx = 0
        self.samples = self.valSamples
        random.shuffle(self.samples)
        
    def getIteratorInfo(self):
        "current batch index and overall number of batches"
        return (self.currIdx // self.batchSize+1, len(self.samples) // self.batchSize)
    
    def hasNext(self):
        "iterator"
        return self.currIdx + self.batchSize <= len(self.samples)
    
    def getNext(self):
        "iterator"
        batchEnd = min(self.currIdx + self.batchSize, len(self.samples))
        batchRange = range(self.currIdx, batchEnd)
        imgPaths = [self.samples[i].imgPath for i in batchRange]
        imgs = [cv2.imread(imgPath) for imgPath in imgPaths]
        imgs = [self.preprocess(img) for img in imgs]
        data = pd.DataFrame()
        for i in batchRange:
            data = data.append(self.samples[i].data,ignore_index=True)

        self.currIdx += self.batchSize
        return Batch(imgPaths, imgs, data, batchRange)
        
    def resize(self, img):
        return cv2.resize(img,self.imgSize)
    
    def bgr2rgb(self, img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    def normalize(self, img):
        return img/255
    
    def white2black(self,img):
        img[np.where((img == [255,255,255]).all(axis = 2))] = [0,0,0]
        return img
    
    def bgr2gray(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    def bgr2rgb(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    def preprocess(self, img):
        #img = self.white2black(img)
        img = self.resize(img)
        #img = self.bgr2rgb(img)
        #img = self.bgr2gray(img)
        img = self.normalize(img)
        return img
    
    def getData(self,dataPath):
        df = pd.read_excel(dataPath,sheet_name='v4')
        df['MF'] = df['Sex'].map(lambda x: 0 if x == 'Male' else 1)
        df['MF'] = df.apply(lambda x: 0.5 if x['Sex'] not in ['Male','Female'] else x['MF'],axis=1)
        return df