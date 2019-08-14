import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
import torchvision
import torch.utils.data as data

from PIL import Image
from glob import glob
from tqdm import tqdm_notebook as tqdm
from skimage import img_as_float


class genericDatagetter(data.Dataset):
    def __init__(self):
        #self.train_samples, self.valid_samples, self.test_samples = self.make_dataset()

        self.loader = cv2_loader
        #self.trainSet()

    def compose_transforms(self, transforms): 
        from experimentUtils import get_transform_called
        if not len(transforms):
            return None
        transform_functions = []
        for transform in transforms:
            if len(transform) == 2:
                name, params = transform
            else: 
                name = transform
                params = dict()
            transform_functions += [get_transform_called(name, params=params)]
        composed_transforms = torchvision.transforms.Compose(transform_functions)
        return composed_transforms
    
    def trainSet(self):
        self.samples = self.train_samples
        self.targets = [s[1] for s in self.train_samples]
        
    def validSet(self):
        self.samples = self.valid_samples
        self.targets = [s[1] for s in self.valid_samples]
        
    def __getitem__(self, index):
        sample = None
        target = None
        _id = None
        return sample, target, _id

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

    
    def make_dataset(self):
        train_images = []
        valid_images = []
        test_images = []
        return train_images, valid_images, test_images

class mnist(genericDatagetter):  
    def __init__(self, root = '/home/rob/Dropbox/thesis/2. code/datasets/mnist', 
                 dataFilePath = '/home/rob/Dropbox/thesis/2. code/src/data/mnist.csv', 
                 transforms=[('resize', {'size': (160, 240)}), 
                          ('rotateandscale',{'rotation':1, 'scaling':0.1, 'background':'white'}),
                          'totensor',
                        ], 
                 background='black', 
                 classifier_column='target', 
                train_subset = 60000,
                test_subset = 10000,
                included_classes = [0,1,2,3,4,5,6,7,8,9]):
        
        self.dataFilePath = dataFilePath
        self.train_subset = train_subset
        self.test_subset = test_subset
        self.included_classes = included_classes
        
        self.train_samples, self.valid_samples, self.test_samples = self.make_dataset(root, dataFilePath, background=background, classifier_column = classifier_column)

        assert len(self.train_samples) > 0, f"No images found in {root}"

        self.root = root
        self.loader = cv2_loader#pil_loader

        self.transforms = transforms
        self.transform = self.compose_transforms(transforms)
        self.target_transform = None
        
        self.trainSet()
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, _id = self.samples[index]
        sample = self.loader(path)
            
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, _id
    
    def make_dataset(self, root, dataFilePath, background='white', classifier_column='target'):
        
        assert background in ['black']
        if background == 'black':
            subfolder = '8_black'
            
        trainImages = []
        validImages = []
        testImages = []
        
        df = pd.read_csv(dataFilePath,sep=';')
        df = df[df['target'].map(lambda x: x in self.included_classes)]
        train_df = df[df['train_test'] == 'train']
        test_df = df[df['train_test'] == 'test']
        train_indices = list(range(len(train_df)))
        np.random.shuffle(train_indices)
        train_indices = train_indices[:self.train_subset]
        test_indices = list(range(len(test_df)))
        np.random.seed(42)
        np.random.shuffle(test_indices)
        test_indices = test_indices[:self.test_subset]
        train_ids = train_df['ID (number)'].values[train_indices]
        train_targets = train_df['target'].values[train_indices]
        test_ids = test_df['ID (number)'].values[test_indices]
        test_targets = test_df['target'].values[test_indices]

        for i in range(len(train_ids)):
            _id = train_ids[i]
            path = f'{root}/{subfolder}/train/{_id}.png'
            target = train_targets[i]
            image = [path, str(target), _id]
            trainImages += [image]
        
        for i in range(len(test_ids)):
            _id = test_ids[i]
            path = f'{root}/{subfolder}/test/{_id}.png'
            target = test_targets[i]
            image = [path, str(target), _id]
            validImages += [image]
        
        return trainImages, validImages, testImages

class toy(genericDatagetter):  
    def __init__(self, root, dataFilePath, classifier_column, transforms=[], datasets=['international','aarhus','copenhagen'], sides='both', background='white'):
        
        self.dataFilePath = dataFilePath
        self.composed_transforms = self.compose_transforms(transforms)
        
        self.train_samples, self.valid_samples, self.test_samples = self.make_dataset(root, dataFilePath, classifier_column, datasets=datasets, sides=sides, background=background)

        assert len(self.train_samples) > 0, f"No images found in {root}"

        self.root = root
        self.loader = cv2_loader#pil_loader
        self.sides = sides

        self.transform = transform
        self.target_transform = None
        
        self.trainSet()
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.sides == 'both_in_one':
            D_path, V_path, target, _id = self.samples[index]
            D_sample = self.loader(D_path)
            V_sample = self.loader(V_path)
            
            sample = np.concatenate([D_sample,V_sample], axis=2)
            sample = self.transform(sample)
        else:
            path, target, _id = self.samples[index]
            sample = self.loader(path)
            
            if self.transform is not None:
                sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, _id
    
    def make_dataset(self, root, dataFilePath, classifier_column, datasets=['aarhus','international','copenhagen'], sides='both', background='white'):
        
        assert background in ['white','neutral']
        if background == 'white':
            subfolder = '4_color_corrected'
        elif background == 'neutral':
            subfolder = '7_neutral'
            
        SEED = 10
        trainSplit = 0.7
        validSplit = 0.85

        images = []

        imgPaths = []
        for dataset in datasets:
            if sides == 'both':
                imgPaths += glob(f'{root}/{dataset}/{subfolder}/*.jpg')
            elif sides == 'both_in_one':
                imgPaths += glob(f'{root}/{dataset}/{subfolder}/*_D.jpg')
            else:
                imgPaths += glob(f'{root}/{dataset}/{subfolder}/*_{sides}.jpg')

        df = pd.read_excel(dataFilePath,sheet_name='v4')

        for path in imgPaths:
            _id = int(os.path.basename(path).split('_')[0])
            target = df.loc[df['ID (number)'] == _id,classifier_column].iloc[0]
            if sides == 'both_in_one':
                V_path = path.replace('_D','_V')
                if os.path.exists(V_path):
                    image = [path, V_path, target, _id]
                else:
                    continue
            else:
                image = [path, target, _id]
            images += [image]

        random.seed(SEED)
        random.shuffle(images)
        trainSplitIdx = int(trainSplit*len(images))
        validSplitIdx = int(validSplit*len(images))
        train_images = images[:trainSplitIdx]
        valid_images = images[trainSplitIdx:validSplitIdx]
        test_images = images[validSplitIdx:]

        return train_images, valid_images, test_images


class butterfly(genericDatagetter):  
    def __init__(self, root, dataFilePath, classifier_column, transforms=[], datasets=['international','aarhus','copenhagen'], sides='both', background='white'):
        
        self.dataFilePath = dataFilePath
        
        self.train_samples, self.valid_samples, self.test_samples = self.make_dataset(root, dataFilePath, classifier_column, datasets=datasets, sides=sides, background=background)

        assert len(self.train_samples) > 0, f"No images found in {root}"

        self.root = root
        self.loader = cv2_loader#pil_loader
        self.sides = sides
        assert sides in ['D','V','both','both_in_one']

        self.transforms = transforms
        self.transform = self.compose_transforms(transforms)
        self.target_transform = None
        
        self.trainSet()
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.sides == 'both_in_one':
            D_path, V_path, target, _id = self.samples[index]
            D_sample = self.loader(D_path)
            V_sample = self.loader(V_path)
            
            sample = np.concatenate([D_sample,V_sample], axis=2)
            sample = self.transform(sample)
        else:
            path, target, _id = self.samples[index]
            sample = self.loader(path)
            
            if self.transform is not None:
                sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, _id
    
    def make_dataset(self, root, dataFilePath, classifier_column, datasets=['aarhus','international','copenhagen'], sides='both', background='white'):
        
        assert background in ['white','neutral']
        if background == 'white':
            subfolder = '4_color_corrected'
        elif background == 'neutral':
            subfolder = '7_neutral'
            
        SEED = 10
        trainSplit = 0.7
        validSplit = 0.85

        images = []

        imgPaths = []
        for dataset in datasets:
            if sides == 'both':
                imgPaths += glob(f'{root}/{dataset}/{subfolder}/*.jpg')
            elif sides == 'both_in_one':
                imgPaths += glob(f'{root}/{dataset}/{subfolder}/*_D.jpg')
            else:
                imgPaths += glob(f'{root}/{dataset}/{subfolder}/*_{sides}.jpg')

        df = pd.read_excel(dataFilePath,sheet_name='v4')

        for path in imgPaths:
            _id = int(os.path.basename(path).split('_')[0])
            target = df.loc[df['ID (number)'] == _id,classifier_column].iloc[0]
            if sides == 'both_in_one':
                V_path = path.replace('_D','_V')
                if os.path.exists(V_path):
                    image = [path, V_path, target, _id]
                else:
                    continue
            else:
                image = [path, target, _id]
            images += [image]

        random.seed(SEED)
        random.shuffle(images)
        trainSplitIdx = int(trainSplit*len(images))
        validSplitIdx = int(validSplit*len(images))
        train_images = images[:trainSplitIdx]
        valid_images = images[trainSplitIdx:validSplitIdx]
        test_images = images[validSplitIdx:]

        return train_images, valid_images, test_images

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def cv2_loader(path):
    return img_as_float(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))