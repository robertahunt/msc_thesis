import os
import torch
import random
import numpy as np
import pandas as pd
import torch.utils.data as data

from PIL import Image
from glob import glob

class butterflyDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, datasets=['international','aarhus','copenhagen'],sides='both'):
        dataFilePath = '/home/rob/Dropbox/thesis/2. code/src/data/data.xlsx'
        self.train_samples, self.valid_samples, self.test_samples = make_dataset(root, dataFilePath, datasets=datasets, sides=sides)

        assert len(self.train_samples) > 0, f"No images found in {root}"

        self.root = root
        self.loader = pil_loader
        self.sides = sides

        self.transform = transform
        self.target_transform = target_transform
        
        self.trainSet()

    def trainSet(self):
        self.samples = self.train_samples
        self.targets = [s[1] for s in self.train_samples]
        
    def validSet(self):
        self.samples = self.valid_samples
        self.targets = [s[1] for s in self.valid_samples]
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.sides == 'both_in_one':
            D_path, V_path, target = self.samples[index]
            D_sample = self.loader(D_path)
            V_sample = self.loader(V_path)
            
            if self.transform is not None:
                D_sample = self.transform(D_sample)
                V_sample = self.transform(V_sample)
            sample = torch.cat((D_sample,V_sample), dim=0)
        else:
            path, target = self.samples[index]
            sample = self.loader(path)
            
            if self.transform is not None:
                sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    
def make_dataset(root, dataFilePath, datasets=['aarhus','international','copenhagen'],sides='both'):
    SEED = 10
    trainSplit = 0.7
    validSplit = 0.85
    
    images = []
    
    imgPaths = []
    for dataset in datasets:
        if sides == 'both':
            imgPaths += glob(f'/home/rob/thesis/datasets/{dataset}/4_color_corrected/*.jpg')
        elif sides == 'both_in_one':
            imgPaths += glob(f'/home/rob/thesis/datasets/{dataset}/4_color_corrected/*_D.jpg')
        else:
            imgPaths += glob(f'/home/rob/thesis/datasets/{dataset}/4_color_corrected/*_{sides}.jpg')

    df = pd.read_excel(dataFilePath,sheet_name='v4')
    df['MF'] = df['Sex'].map(lambda x: 0 if x == 'Male' else 1)
    df['MF'] = df.apply(lambda x: 0.5 if x['Sex'] not in ['Male','Female'] else x['MF'],axis=1)
            
    for path in imgPaths:
        _id = int(os.path.basename(path).split('_')[0])
        target = df.loc[df['ID (number)'] == _id,'Sex'].iloc[0]
        if sides == 'both_in_one':
            V_path = path.replace('_D','_V')
            if os.path.exists(V_path):
                images += [(path, V_path, target)]
        else:
            images += [(path, target)]
        
    random.seed(SEED)
    random.shuffle(images)
    trainSplitIdx = int(trainSplit*len(images))
    validSplitIdx = int(validSplit*len(images))
    train_images = images[:trainSplitIdx]
    valid_images = images[trainSplitIdx:validSplitIdx]
    test_images = images[validSplitIdx:]
    
    return train_images, valid_images, test_images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')