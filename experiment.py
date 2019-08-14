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
from torch import autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm_notebook as tqdm
from shutil import move
from pprint import pprint
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

from models.BaseModel import BaseModel
from experimentUtils import get_model_called, get_loss_function_called, get_optimizer_called, get_datagetter_called, get_transform_called
from utils import check_memory_usage, start_timer, tick, count_parameters

def load_experiment(filepath):
    assert os.path.isfile(filepath), f"Cannot find file  {filepath}"
    print(f"=> loading checkpoint '{filepath}'")
    config = torch.load(filepath)['configuration']
    pprint(config)
    ex = experiment(**config)
    ex.prep_experiment()
    ex.load(filepath)
    return ex

class experiment:
    def __init__(self,
                 model = None,
                 modelName='',
                 modelParams = dict(),
                 loss = 'mse',
                 opt = 'adam',
                 optParams = dict(lr=0.0001),
                 imgSize = (240,160),
                 batchSize = 30,
                 earlyStopping = 5,
                 max_num_epochs = 100,
                 denoise = False,#('gaussian_noise',dict(sigma=1))
                 cuda = True,
                 suffix = '',
                 save_me = True,
                 datagetter_name = 'butterfly',
                 datagetterParams = dict(
                     background = 'white', 
                     dataFilePath = '/home/rob/Dropbox/thesis/2. code/src/data/data.xlsx',
                     root = '/home/rob/Dropbox/thesis/2. code/datasets',
                     classifier_column = 'Sex',
                     transforms = [('resize',dict(size=(160,240))),
                                   ('hflip'),
                                   ('totensor')],
                     sides = 'both',
                     datasets = ['international','copenhagen','aarhus'],
                 )
                ):
        
        self.modelFolder = '/home/rob/Dropbox/thesis/2. code/src/experiments'
        assert os.path.exists(self.modelFolder), "Cannot find model folder, abort!"
        self.datagetter_name = datagetter_name
        self.datagetterParams = datagetterParams
        
        self.model = model
        if model:
            self.modelName = model.__class__.__name__
        else:
            self.modelName = modelName
        
        self.tmp_save_fp = os.path.join(self.modelFolder,self.modelName+suffix+ '.pth.tar')
        self.modelParams = modelParams
        
        self.loss = loss
        if 'sides' in datagetterParams.keys():
            self.sides = datagetterParams['sides']
        else:
            self.sides = None
        
        self.opt = opt
        self.optParams = optParams
        
        self.imgSize = imgSize
        self.batchSize = batchSize
        
        self.earlyStopping = earlyStopping
        self.max_num_epochs = max_num_epochs
            
        self.denoise = denoise
        
        self.cuda = cuda
        self.suffix = suffix
        self.save_me = save_me
        self.date = pd.Timestamp.now().date()
        
        self.results = dict(
            epoch_durations = [],
            date = self.date,
            best_val_loss = float('inf'),
            train_losses = [],
            valid_losses = [],
            sep_train_losses = [],
            sep_valid_losses = [],
            classifier_t_accuracies = [],
            classifier_v_accuracies = [],
            classifications_t = [],
            classifications_v = [],
            trainable_parameters = None,
            outputs_example = None
        )
        
        self.prep_experiment()
        
    def save(self, filepath):
        self.net.save({
                'configuration':self.configuration,
                'results':self.results,
                'state_dict':self.net.state_dict(),
                'optimizer_dict':self.optimizer.state_dict()},filepath)
        
    def load(self, filepath):
        if os.path.isfile(filepath):
            print(f"=> loading checkpoint '{filepath}'")
            checkpoint = torch.load(filepath)
            self.net.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_dict'])
            self.results = checkpoint['results']
            print('Checkpoint results:')
            self.print_results()
            if 'p' in checkpoint:
                self.net.p = checkpoint['p']
            
            print(f"=> loaded checkpoint '{filepath}'")
        else:
            print(f"=> no checkpoint found at '{filepath}'")
            
    
    def print_results(self):
        res = self.results
        print('Avg Epoch Duration: %s'%np.mean(res['epoch_durations']))
        print('Date: %s'%res['date'])
        print('Best Validation Loss: %s'%res['best_val_loss'])
        print('Latest Train Loss: %s'%res['train_losses'][-1])
        print('Latest Valid Loss: %s'%res['valid_losses'][-1])
    
    @property
    def configuration(self):
        return dict(
            modelName = self.modelName,
            modelParams = self.modelParams,
            loss = self.loss,
            opt = self.opt,
            optParams = self.optParams,
            imgSize = self.imgSize,
            batchSize = self.batchSize,
            earlyStopping = self.earlyStopping,
            max_num_epochs = self.max_num_epochs,
            cuda = self.cuda,
            denoise = self.denoise,
            datagetter_name = self.datagetter_name,
            datagetterParams = self.datagetterParams
        )
    
    
    def add_noise(self, x):
        name, params = self.denoise
        transform = get_transform_called(name, params=params)
        return transform(x)
        
    def prep_experiment(self):
        print('Prepping Experiment..')
        self.datagetter = get_datagetter_called(self.datagetter_name, params=self.datagetterParams)
        self.loader = DataLoader(self.datagetter, batch_size=self.batchSize, shuffle=True, num_workers=4)
        if self.model:
            self.net = self.model
        else:
            self.net = get_model_called(self.modelName, params=self.modelParams)
            
        if self.cuda:
            self.net = self.net.cuda()
        self.results['trainable_parameters'] = count_parameters(self.net);
        self.loss_function = get_loss_function_called(self.loss)
        self.optimizer = get_optimizer_called(self.opt, netparams=self.net.parameters(), hparams=self.optParams)
        
    def test_experiment(self):
        print('Testing data throughput...')
        x,y,_id = next(iter(self.loader))
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
            #print('batch: %s of %s'%(batch_num, numBatches))
            x, y, _id = batch
            batchSize = x.size()[0]
            if self.cuda:
                x = x.cuda()
                
            self.optimizer.zero_grad()
            reset_prior = True if batch_num == 0 else False
            if self.denoise:
                noisy_x = self.add_noise(x)
                outputs = self.net(noisy_x, reset_prior) if hasattr(self.net, 'reset_prior') else self.net(noisy_x)
                outputs['noisy_x'] = noisy_x
            else:
                outputs = self.net(x, reset_prior) if hasattr(self.net, 'reset_prior') else self.net(x)
                
            outputs['x'] = x

            loss, ind_loss = self.loss_function(outputs)
            ind_loss = [ls.item() for ls in ind_loss]
            with autograd.detect_anomaly():
                loss.backward()
                self.optimizer.step()

            losses += loss.item()*batchSize

            if batch_num == 0:
                ind_losses = np.multiply(batchSize,ind_loss)
            else:
                ind_losses += np.multiply(batchSize,ind_loss)
                
        avg_loss = losses/numImages
        avg_ind_loss = ind_losses/numImages
        return avg_loss, avg_ind_loss
        
    def validate(self, run_one_batch = False):
        # Evaluate, do not propagate gradient
        self.net.eval()
        self.net.train(False)
        self.loader.dataset.validSet()
        numBatches = len(self.loader)
        numImages = len(self.loader.dataset)
        
        losses = 0
        with torch.no_grad():
            for batch_num, batch in enumerate(self.loader):
                # Just load a single batch from the test loader
                x, y, _id = batch
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
                
                loss, ind_loss = self.loss_function(outputs)
                ind_loss = [ls.item() for ls in ind_loss]
                
                losses += loss.item()*batchSize
                
                if batch_num == 0:
                    ind_losses = np.multiply(batchSize,ind_loss)
                else:
                    ind_losses += np.multiply(batchSize,ind_loss)
                
                if run_one_batch == True:
                    break
        if run_one_batch == True:
            avg_loss = losses/batchSize
            avg_ind_losses = ind_losses/batchSize
        else:
            avg_loss = losses/numImages
            avg_ind_losses = ind_losses/numImages
                
        return batch, outputs, avg_loss, avg_ind_losses
    
    def get_latent_space(self, _set = 'train'):
        assert _set in ['train','valid']
        self.net.eval()
        self.net.train(False)
        self.net = self.net.cpu()
        
        #transforms = [('resize',dict(size=(self.imgSize[1],self.imgSize[0]))), ('totensor')]
        classifier_datagetterParams = self.datagetterParams.copy()
        #classifier_datagetterParams['transforms'] = transforms
        

        datagetter = get_datagetter_called(self.datagetter_name, params=classifier_datagetterParams)
        loader = DataLoader(datagetter, batch_size=self.batchSize, shuffle=True, num_workers=4)
        if hasattr(self.net, '_cuda'):
            self.net._cuda = False
        if _set == 'train':
            loader.dataset.trainSet()
        else:
            loader.dataset.validSet()
        ys = []
        _ids = []
        
        for batch_num, batch in tqdm(enumerate(loader), desc=f'Getting latent space for {_set} set'):  
            x, y, _id = batch
            batchSize = x.size()[0]
            
            self.optimizer.zero_grad()
            
            if self.denoise:
                noisy_x = self.add_noise(x)
                z, _ = self.net.encoder(noisy_x)
            else:
                z, _ = self.net.encoder(x)
                
            if hasattr(self.net, 'take_samples'):
                z, _, _, _, _, _, _ = self.net.take_samples(z, batchSize)
            
            num_samples = int(z.shape[0]/len(y))
            y = np.expand_dims(np.array(y),1).repeat(num_samples,axis=1)
            y = y.flatten()
            
            if batch_num == 0:
                zs = z.detach().numpy()
            else:
                zs = np.append(zs, z.detach().numpy(),axis=0)
            #_ids += list(_id)
            ys += list(y)
        return zs, ys, _ids

    
    def get_full_latent_space(self):
        #get latent space
        #and classificaiton values
        #for train and validation seperately
        z_train, y_train, train_ids = self.get_latent_space(_set='train')
        z_valid, y_valid, valid_ids = self.get_latent_space(_set='valid')
        
        return z_train, y_train, train_ids, z_valid, y_valid, valid_ids
        
    def make_classifier(self, z_train_shape, output_classes):
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
        classifier_net = FFNClassifier(input_shape=z_train_shape,hidden_units=100,output_classes=output_classes)
        if classifier_cuda:
            classifier_net = classifier_net.cuda()

        return classifier_net
    
    def run_classifier(self):
        #self.z_train, self.y_train, self.train_ids, self.z_valid, self.y_valid, self.valid_ids = self.get_full_latent_space()
        z_train, y_train, train_ids, z_valid, y_valid, valid_ids = self.get_full_latent_space()
        output_classes = len(np.unique(y_train))
        classifier_net = self.make_classifier(z_train.shape, output_classes)
        classifier_optimizer = optim.Adam(classifier_net.parameters(), lr=0.001)
        classifier_loss = nn.CrossEntropyLoss()
    
        #z_train = Variable(torch.from_numpy(self.z_train))
        y_train = y_train
        #z_valid = Variable(torch.from_numpy(self.z_valid))
        y_valid = y_valid
        z_train = torch.as_tensor(z_train)
        z_valid = torch.as_tensor(z_valid)
       
        label_encoder = LabelEncoder()
        label_encoder.fit(y_train)

        max_epochs = 10000
        earlyStopping = 20
        best_val_loss = np.inf
        noImprovement = 0
        c_train_losses = []
        c_val_losses = []
        for e in range(max_epochs):
            #print('e:',e)
            losses = []
            classifier_net.train()

            #train_y = torch.as_tensor(label_encoder.transform(y_train))
            train_y = Variable(torch.from_numpy(label_encoder.transform(y_train)))
            
            train_preds = classifier_net(z_train)
            loss = classifier_loss(train_preds, train_y)

            classifier_optimizer.zero_grad()
            loss.backward()
            classifier_optimizer.step()
            #c_train_losses += [loss]

            losses = []
            classifier_net.eval()

            #valid_y = torch.as_tensor(label_encoder.transform(y_valid))
            valid_y = Variable(torch.from_numpy(label_encoder.transform(y_valid)))

            valid_preds = classifier_net(z_valid)
            val_loss = classifier_loss(valid_preds, valid_y)
            #c_val_losses += [val_loss]
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
        
        return z_train.numpy(), y_train, z_valid.numpy(), y_valid, train_accuracy, valid_accuracy, e
        
        
    def run_experiment(self, n_arand_clusters, run_one_batch=False):
        print('Running Experiment with Configuration: ')
        pprint(self.configuration)
        
        #Ensure net is on cuda
        if self.cuda:
            self.net = self.net.cuda()
            self.net._cuda = True
        
        noImprovementSince = 0
        epochs = len(self.results['train_losses'])
        assert self.max_num_epochs > epochs, f"Error, max_num_epochs ({self.max_num_epochs}) must be greater than epochs already run ({epochs})."
        
        if run_one_batch == True:
            batch, outputs, valid_loss, sep_valid_loss = self.validate(run_one_batch = True)
            epoch = epochs
        else:
            for epoch in range(epochs, self.max_num_epochs):
                start_time = pd.Timestamp.now()
                train_loss, sep_train_loss = self.train()
                self.results['train_losses'].append(train_loss)
                self.results['sep_train_losses'].append(sep_train_loss)

                batch, outputs, valid_loss, sep_valid_loss= self.validate()
                self.results['valid_losses'].append(valid_loss)
                self.results['sep_valid_losses'].append(sep_valid_loss)
                print(sep_valid_loss)
                self.results['epoch_durations'] += [pd.Timestamp.now() - start_time]
                
                
                if valid_loss < self.results['best_val_loss']:
                    self.results['best_val_loss'] = valid_loss
                    noImprovementSince = 0
                    print('Model Improved, saving model.')
                    self.save(self.tmp_save_fp)
                else:
                    noImprovementSince += 1
                    print(f'No improvement on validation set for {noImprovementSince} epochs.')

                if epoch == 0:
                    continue

                print(epoch, '/', self.max_num_epochs)

                self.net.plot(self.results['train_losses'],self.results['valid_losses'],outputs,batch,batch[0].size()[0], cuda=self.cuda, sides=self.sides)
                if noImprovementSince >= self.earlyStopping:
                    print(f'No improvement on validation set for {self.earlyStopping} epochs. Quiting.')
                    break
                    
                
            #load the best last model
            self.load(self.tmp_save_fp)
              
        outputs['batch'] = batch
        self.results['outputs_example'] = outputs
        
        #Free up some gpu memory
        if self.cuda:
            batch[0] = batch[0].cpu()
            outputs['x_hat'] = outputs['x_hat'].cpu()
        torch.cuda.empty_cache()
        gc.collect()
        
        self.results['z_train'], self.results['y_train'], self.results['z_valid'], self.results['y_valid'], self.results['classifier_t_accuracy'], self.results['classifier_v_accuracy'], self.classifier_epochs = self.run_classifier()
        classifier_results = [self.results['z_train'], self.results['y_train'], self.results['z_valid'], self.results['y_valid']]
        
        self.results['classifier_epochs'] = self.classifier_epochs
        if self.save_me:
            self.save(self.tmp_save_fp)
            arand = self.calc_arand(n_arand_clusters)
            newModelName = '%1.2f_'%arand + '%1.2f_'%self.results['classifier_v_accuracy'] + '%1.2f'%self.results['classifier_t_accuracy'] + f'_{self.loss}_%06.0f'%(self.results['best_val_loss']*100000) + f'_{str(self.date)}_{self.modelName}{self.suffix}' 
            new_fp = os.path.join(self.modelFolder,newModelName+'.pth.tar')
            move(self.tmp_save_fp,new_fp)  

            plotfp = os.path.join(self.modelFolder,'plots',newModelName+'.html')
            self.net.plot(self.results['train_losses'],self.results['valid_losses'],outputs,batch, batch[0].size()[0], cuda=False, results=classifier_results, savefp=plotfp, sides=self.sides)
        else: 
            self.net.plot(self.train_losses,self.valid_losses,outputs,batch, batch[0].size()[0], cuda=False, results=classifier_results, sides=self.sides)
            
    def calc_arand(self, n_clusters):
        num_samples = 1
        n_train, d = self.results['z_train'].shape
        n_valid, d = self.results['z_valid'].shape

        sample_number = 0
        z_clust = self.results['z_valid'].reshape(-1,num_samples,d)[:,sample_number,:].copy()
        y_clust = np.array(self.results['y_valid']).reshape(-1,num_samples)[:,sample_number].copy()    
    
        clust = KMeans(n_clusters=n_clusters)
        y_valid_pred = clust.fit_predict(z_clust)
        score = adjusted_rand_score(y_clust, y_valid_pred)
        return score
        
        