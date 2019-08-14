import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus

import numpy as np

from sklearn.manifold import TSNE
from .BaseModel import BaseModel
from .modelUtils import calc_size_after_mp, calc_size_after_conv, make_random_GMM, start_timer, tick, initialize_gmm
    
class Encoder(BaseModel):
    def __init__(self, imgSize, n_conv_layers, filter_size, n_filters, in_channels, stride=1):
        super(Encoder, self).__init__()
        self.imgSize = imgSize
        self.in_channels = in_channels
        self.n_conv_layers = n_conv_layers
        self.filter_size = filter_size
        self.padding = filter_size//2
        self.stride = stride
        self.n_filters = n_filters
        self.c0 = nn.Conv2d(in_channels=in_channels, out_channels=n_filters, kernel_size=filter_size, padding=self.padding, stride=stride)
        self.sizes = [calc_size_after_conv(imgSize, filter_size, stride=stride, padding=self.padding)]
        for i in range(1,n_conv_layers):
            exec(f"self.c{i} = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=filter_size, padding=self.padding, stride=stride)")      
            self.sizes += [calc_size_after_conv(self.sizes[-1], filter_size, stride=stride, padding=self.padding)]
        
        self.l0 = nn.Linear(in_features = np.prod(self.sizes[-1])*n_filters, out_features = 30)
        self.l1 = nn.Linear(in_features = 30, out_features = 30)
        self.l2 = nn.Linear(in_features = 30, out_features = 30)
        
    def forward(self, x):
        import torch.nn.functional as F
        batch_size = x.shape[0]
        #x = x.view(batch_size, self.in_channels, self.imgSize[0], self.imgSize[1])
        for i in range(self.n_conv_layers):
            x = eval(f"F.relu(self.c{i}(x))")
        x = x.view(batch_size, np.prod(self.sizes[-1])*self.n_filters)
        for i in range(2):
            x = eval(f"F.relu(self.l{i}(x))")
        x = eval(f"self.l2(x)")
        return x, None
    
class Decoder(BaseModel):
    def __init__(self, imgSize, n_conv_layers, filter_size, n_filters, out_channels, sizes, stride=1):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.imgSize = imgSize
        self.n_conv_layers = n_conv_layers
        self.filter_size = filter_size
        self.padding = filter_size//2
        self.stride = stride
        self.n_filters = n_filters
        self.sizes = sizes
        
        self.l2 = nn.Linear(in_features = 30, out_features = 30)
        self.l1 = nn.Linear(in_features = 30, out_features = 30)
        self.l0 = nn.Linear(in_features = 30, out_features = np.prod(sizes[-1])*n_filters)

        
        for i in range(n_conv_layers-1,0,-1):
            exec(f"self.c{i} = nn.ConvTranspose2d(in_channels=n_filters, out_channels=n_filters, kernel_size=filter_size, padding=self.padding, stride=stride, )") 
        self.c0 = nn.ConvTranspose2d(in_channels=n_filters, out_channels=out_channels, kernel_size=filter_size, padding=self.padding, stride=stride)
        
    def forward(self, x):
        batch_size = x.shape[0]
        for i in range(2,-1,-1):
            x = eval(f"self.l{i}(F.relu(x))")
        x = x.view(batch_size, self.n_filters, self.sizes[-1][0], self.sizes[-1][1])
        for i in range(self.n_conv_layers-1,0,-1):
            x = eval(f"self.c{i}(F.relu(x),output_size=self.sizes[{i-1}])")
        
        x = self.c0(F.relu(x),output_size=self.imgSize)
        x = x.view(batch_size, self.out_channels, self.imgSize[0], self.imgSize[1])
        return x
    

class AEMnist3Conv(BaseModel):
    def __init__(self, imgSize, n_conv_layers, filter_size, n_filters, out_channels, cuda, stride=1):
        super(AEMnist3Conv, self).__init__()
        self._cuda = cuda
        self.imgSize=imgSize
        self.in_channels = out_channels
        self.out_channels = out_channels
        self.n_conv_layers = n_conv_layers
        self.filter_size = filter_size
        self.padding = filter_size//2
        self.stride = stride
        self.n_filters = n_filters
        
        
        # We encode the data onto the latent space using two linear layers
        self.encoder = Encoder(imgSize, n_conv_layers, filter_size, n_filters, self.in_channels, stride=stride)
        # The latent code must be decoded into the original image
        self.decoder = Decoder(imgSize, n_conv_layers, filter_size, n_filters, out_channels, sizes = self.encoder.sizes, stride=stride)
        self.device = torch.device("cuda:0" if cuda else "cpu")
        
        
    def forward(self, x): 
        batch_size = x.shape[0]
        outputs = {'x':x}
        
        # Split encoder outputs into a mean and variance vector
        z, _ = self.encoder(x) #z shape: [batch_size, 2*latent_features + n_gaussians]
        #split z into mus/vars and gaussian weights, w
        
        x = self.decoder(z)
        # The original digits are on the scale [0, 1]
        x = torch.sigmoid(x)
        
        # Mean over samples      
        #print(mu.shape, mus.shape, sigmas.shape) 
        outputs["x_hat"] = x
        outputs["z"] = z
        outputs["cuda"] = self._cuda
        
        return outputs
    
    