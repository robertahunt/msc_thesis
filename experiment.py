import os
import gc
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from shutil import move
from pprint import pprint
from sklearn.preprocessing import LabelEncoder

from models.BaseModel import BaseModel
from butterflyDataset import butterflyDataset
from experimentUtils import get_model_called, get_loss_function_called, get_optimizer_called, get_transform_called
from utils import plot, check_memory_usage, start_timer, tick, count_parameters

def load_experiment(filepath):
    assert os.path.isfile(filepath), f"Cannot find file  {filepath}"
    print(f"=> loading checkpoint '{filepath}'")
    checkpoint = torch.load(filepath)
    print(checkpoint['configuration'])
    ex = experiment(**checkpoint['configuration'])
    ex.prep_experiment()
    ex.net.load_state_dict(checkpoint['state_dict'])
    ex.optimizer.load_state_dict(checkpoint['optimizer_dict'])

    print(f"=> loaded checkpoint '{filepath}'")
    return ex

class experiment:
    def __init__(self,
                 model = None,
                 modelName='',
                 modelParams = dict(),
                 loss = 'mse',
                 opt = 'adam',
                 optParams = dict(lr=0.0001),
                 sides = 'both',
                 imgSize = (240,160),
                 batchSize = 30,
                 datasets = ['international','copenhagen','aarhus'],
                 earlyStopping = 5,
                 max_num_epochs = 100,
                 transforms = [('resize',dict(size=(160,240))),
                               ('hflip'),
                               ('totensor')],
                 denoise = False,#('gaussian_noise',dict(sigma=1))
                 cuda = True,
                 suffix = ''):
        
        assert sides in ['D','V','both','both_in_one']
        self.modelFolder = '/home/rob/Dropbox/thesis/2. code/src/experiments'
        assert os.path.exists(self.modelFolder), "Cannot find model folder, abort!"
        self.datasets_root = '/home/rob/Dropbox/thesis/datasets'
            
        self.model = model
        if model:
            self.modelName = model.__class__.__name__
        else:
            self.modelName = modelName
        self.modelParams = modelParams
        
        self.loss = loss
        
        self.opt = opt
        self.optParams = optParams
        
        self.sides = sides
        self.imgSize = imgSize
        self.batchSize = batchSize
        self.datasets = datasets
        
        self.earlyStopping = earlyStopping
        self.max_num_epochs = max_num_epochs
        
        self.transforms = transforms
        self.composed_transforms = self.compose_transforms(self.transforms)
            
        self.denoise = denoise
        
        self.cuda = cuda
        self.suffix = suffix
        self.date = pd.Timestamp.now().date()

        self.configuration = dict(
            modelName = modelName,
            modelParams = modelParams,
            loss = loss,
            opt = opt,
            optParams = optParams,
            sides = sides,
            imgSize = imgSize,
            batchSize = batchSize,
            datasets = datasets,
            earlyStopping = earlyStopping,
            max_num_epochs = max_num_epochs,
            transforms = transforms,
            cuda = cuda,
            denoise = denoise,
        )
        
        self.results = dict(
            avg_epoch_duration = None,
            date = self.date,
            epochs = None,
            best_valid_loss = None,
            train_losses = None,
            valid_losses = None,
            stopped_early = False,
        )
        
        self.prep_experiment()
    
    def add_noise(self, x):
        name, params = self.denoise
        transform = get_transform_called(name, params=params)
        return transform(x)
    
    def compose_transforms(self, transforms):
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
        
    def prep_experiment(self):
        print('Prepping Experiment..')
        self.dataset = butterflyDataset(self.datasets_root, transform=self.composed_transforms, datasets=self.datasets, sides=self.sides)
        self.loader = DataLoader(self.dataset, batch_size=self.batchSize, shuffle=True, num_workers=4)
        if self.model:
            self.net = self.model
        else:
            self.net = get_model_called(self.modelName, params=self.modelParams)
            
        if self.cuda:
            self.net = self.net.cuda()
        self.trainable_parameters = count_parameters(self.net);
        self.loss_function = get_loss_function_called(self.loss)
        self.optimizer = get_optimizer_called(self.opt, netparams=self.net.parameters(), hparams=self.optParams)
        
    def test_experiment(self):
        print('Testing data throughput...')
        x,y = next(iter(self.loader))
        print('Input shape: ',x.shape)
        if self.cuda:
            x = x.cuda()

        outputs = self.net(x)
        print('Output shape: ',outputs['x_hat'].shape)
        assert outputs['x_hat'].shape == x.shape
        print('Test Successful.')
        return outputs
    
    def train(self):
        self.net.train(True)
        self.loader.dataset.trainSet()
        numBatches = len(self.loader)
        numImages = len(self.loader.dataset)
        
        losses = 0
        
        for batch_num, batch in enumerate(self.loader):  
            x, y = batch
            batchSize = x.size()[0]
            if self.cuda:
                x = x.cuda()
                
            self.optimizer.zero_grad()
            
            if self.denoise:
                noisy_x = self.add_noise(x)
                outputs = self.net(noisy_x)
                outputs['noisy_x'] = noisy_x
            else:
                outputs = self.net(x)
                
            outputs['x'] = x

            loss = self.loss_function(outputs)

            loss.backward()
            self.optimizer.step()

            losses += loss.item()*batchSize
            #print(loss.item())
        avg_loss = losses/numImages
        #print('train loss: ',avg_loss)
        return avg_loss
        
    def validate(self):
        # Evaluate, do not propagate gradient
        self.net.eval()
        self.net.train(False)
        self.loader.dataset.validSet()
        numBatches = len(self.loader)
        numImages = len(self.loader.dataset)
        
        losses = 0
        with torch.no_grad():

            avg_loss = 0
            for batch_num, batch in enumerate(self.loader):
                # Just load a single batch from the test loader
                x, y = batch
                batchSize = x.size()[0]

                if self.cuda:
                    x = x.cuda()
                    
                if self.denoise:
                    noisy_x = self.add_noise(x)
                    outputs = self.net(noisy_x)
                    outputs['noisy_x'] = noisy_x
                else:
                    outputs = self.net(x)

                outputs['x'] = x
                
                loss = self.loss_function(outputs)

                losses += loss.item()*batchSize
        avg_loss = losses/numImages
        #print('valid loss: ',avg_loss)
                
        return batch, outputs, avg_loss
    
    def get_latent_space(self, _set = 'train'):
        assert _set in ['train','valid']
        self.net.eval()
        self.net.train(False)
        self.net = self.net.cpu()
        
        transforms = [('resize',dict(size=(160,240))), ('totensor')]
        composed_transforms = self.compose_transforms(transforms)
        dataset = butterflyDataset(self.datasets_root, transform=composed_transforms, datasets=self.datasets, sides=self.sides)
        loader = DataLoader(dataset, batch_size=self.batchSize, shuffle=True, num_workers=4)
        if hasattr(self.net, '_cuda'):
            self.net._cuda = False
        if _set == 'train':
            loader.dataset.trainSet()
        else:
            loader.dataset.validSet()
        ys = []
        
        for batch_num, batch in enumerate(loader):  
            x, y = batch
            batchSize = x.size()[0]
            #if self.cuda:
            #    x = x.cuda()
                
            self.optimizer.zero_grad()
            
            if self.denoise:
                noisy_x = self.add_noise(x)
                z, _ = self.net.encoder(noisy_x)
            else:
                z, _ = self.net.encoder(x)
                
            if batch_num == 0:
                zs = z.detach().numpy()
            else:
                zs = np.append(zs, z.detach().numpy(),axis=0)
            ys += list(y)
        return zs, ys

    
    def get_full_latent_space(self):
        #get latent space
        #and classificaiton values
        #for train and validation seperately
        z_train, y_train = self.get_latent_space(_set='train')
        z_valid, y_valid = self.get_latent_space(_set='valid')
        
        return z_train, y_train, z_valid, y_valid
        
    def make_classifier(self, z_train_shape):
        classifier_cuda = False

        class FFNClassifier(BaseModel):
            def __init__(self,input_shape,hidden_units,output_classes):
                super(FFNClassifier, self).__init__()
                self.batchSize = input_shape[0]
                self.num_features = np.prod(input_shape[1:])

                self.classifier = nn.Sequential(
                    nn.Linear(in_features=self.num_features,out_features=hidden_units),
                    nn.ReLU(),
                    nn.Linear(in_features=hidden_units,out_features=output_classes)
                )

            def forward(self, z):
                z = self.classifier(z)
                return torch.softmax(z,dim=1)
        classifier_net = FFNClassifier(input_shape=z_train_shape,hidden_units=100,output_classes=3)
        if classifier_cuda:
            classifier_net = classifier_net.cuda()

        return classifier_net
    
    def run_classifier(self):
        classifier_net = self.make_classifier(self.z_train.shape)
        classifier_optimizer = optim.Adam(classifier_net.parameters(), lr=0.001)
        classifier_loss = nn.CrossEntropyLoss()
    
        z_train = Variable(torch.from_numpy(self.z_train))
        y_train = self.y_train
        z_valid = Variable(torch.from_numpy(self.z_valid))
        y_valid = self.y_valid
        
        label_encoder = LabelEncoder()
        label_encoder.fit(y_train)

        max_epochs = 10000
        earlyStopping = 20
        best_val_loss = np.inf
        noImprovement = 0
        c_train_losses = []
        c_val_losses = []
        for e in range(max_epochs):
            losses = []
            classifier_net.train()

            train_y = Variable(torch.from_numpy(label_encoder.transform(y_train)))

            train_preds = classifier_net(z_train)
            loss = classifier_loss(train_preds, train_y)

            classifier_optimizer.zero_grad()
            loss.backward()
            classifier_optimizer.step()
            c_train_losses += [loss]

            losses = []
            classifier_net.eval()

            valid_y = Variable(torch.from_numpy(label_encoder.transform(y_valid)))

            valid_preds = classifier_net(z_valid)
            val_loss = classifier_loss(valid_preds, valid_y)
            c_val_losses += [val_loss]
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                noImprovement = 0
            else:
                noImprovement += 1
            if noImprovement == earlyStopping:
                print('No classifier improvement, stopping after %s epochs'%e)
                print(best_val_loss)
                
                break
                
        train_accuracy = float((torch.max(train_preds, 1)[1] == train_y).sum())/len(train_y)
        valid_accuracy = float((torch.max(valid_preds, 1)[1] == valid_y).sum())/len(valid_y)
        
        return train_accuracy, valid_accuracy, e
        
        
    def run_experiment(self):
        print('Running Experiment with Configuration: ')
        pprint(self.configuration)
        
        best_val_loss = float('inf')
        noImprovementSince = 0
        train_losses = []
        valid_losses = []
        self.avg_epoch_duration = pd.Timedelta(0)
        
        for epoch in range(self.max_num_epochs):
            start_time = pd.Timestamp.now()
            train_loss = self.train()
            train_losses.append(train_loss)
            
            batch, outputs, valid_loss = self.validate()
            valid_losses.append(valid_loss)

            edur = pd.Timestamp.now() - start_time
            self.avg_epoch_duration = (self.avg_epoch_duration*(epoch) + edur)/(epoch+1)
            #print(self.avg_epoch_duration)
            
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                noImprovementSince = 0
                fp = os.path.join(self.modelFolder,self.modelName+'.pth.tar')
                print('Model Improved, saving model.')
                self.results = dict(
                    avg_epoch_duration = self.avg_epoch_duration,
                    date = self.date,
                    epochs = epoch,
                    best_valid_loss = best_val_loss,
                    train_losses = train_losses,
                    valid_losses = valid_losses,
                    stopped_early = True if epoch == self.max_num_epochs - self.earlyStopping else False,
                    trainable_parameters = self.trainable_parameters
                )
                
                self.net.save({
                    'configuration':self.configuration,
                    'results':self.results,
                    'state_dict':self.net.state_dict(),
                    'optimizer_dict':self.optimizer.state_dict()},fp)
            else:
                noImprovementSince += 1

            if epoch == 0:
                continue

            print(epoch, '/', self.max_num_epochs)
            plot(train_losses,valid_losses,outputs,batch,epoch,batch[0].size()[0], cuda=self.cuda, sides=self.sides)
            if noImprovementSince >= self.earlyStopping:
                print(f'No improvement on validation set for {self.earlyStopping} epochs. Quiting.')
                break
              

        outputs['batch'] = batch
        self.outputs_example = outputs
        
        #Free up some gpu memory
        if self.cuda:
            batch[0] = batch[0].cpu()
            outputs['x_hat'] = outputs['x_hat'].cpu()
            outputs['z'] = outputs['z'].cpu()
        torch.cuda.empty_cache()
        gc.collect()
        
        self.z_train, self.y_train, self.z_valid, self.y_valid = self.get_full_latent_space()
        self.classifier_t_accuracy, self.classifier_v_accuracy, self.classifier_epochs = self.run_classifier()
        
        self.results['classifier_t_accuracy'] = self.classifier_t_accuracy
        self.results['classifier_v_accuracy'] = self.classifier_v_accuracy
        self.results['classifier_epochs'] = self.classifier_epochs
        self.net.save({
            'configuration':self.configuration,
            'results':self.results,
            'state_dict':self.net.state_dict(),
            'optimizer_dict':self.optimizer.state_dict()},fp)
        
        newModelName = '%1.2f_'%self.classifier_v_accuracy + '%1.2f'%self.classifier_t_accuracy + f'_{self.loss}_%06.0f'%(best_val_loss*100000) + f'_{str(self.date)}_{self.modelName}{self.suffix}' 
        new_fp = os.path.join(self.modelFolder,newModelName+'.pth.tar')
        move(fp,new_fp)  
        
        plotfp = os.path.join(self.modelFolder,'plots',newModelName+'.jpg')
        results = [self.z_train, self.y_train, self.z_valid, self.y_valid]
        plot(train_losses,valid_losses,outputs,batch, epoch, batch[0].size()[0], cuda=False, results=results, savefp=plotfp, sides=self.sides)
        
        