import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus

import numpy as np

from sklearn.manifold import TSNE
from .BaseModel import BaseModel
from .modelUtils import calc_size_after_mp, calc_size_after_conv, make_random_GMM, start_timer, tick, initialize_gmm
    
class Encoder(BaseModel):
    def __init__(self, imgSize, n_hidden_layers, hidden_features, latent_features, in_channels):
        super(Encoder, self).__init__()
        self.imgSize = imgSize
        self.in_channels = in_channels
        self.n_hidden_layers = n_hidden_layers
        self.l0 = nn.Linear(in_features = np.prod(imgSize)*in_channels, out_features = hidden_features)
        
        for i in range(1,n_hidden_layers+1):
            exec(f"self.l{i} = nn.Linear(in_features = hidden_features, out_features = hidden_features)")
        
        exec(f"self.l{n_hidden_layers+1} = nn.Linear(in_features = hidden_features, out_features = latent_features)")
        
    def forward(self, x):
        import torch.nn.functional as F
        batch_size = x.shape[0]
        x = x.view(batch_size, np.prod(self.imgSize)*self.in_channels)
        for i in range(self.n_hidden_layers+1):
            x = eval(f"F.relu(self.l{i}(x))")
        x = eval(f"self.l{self.n_hidden_layers+1}(x)")
        return x, None
    
class Decoder(BaseModel):
    def __init__(self, imgSize, n_hidden_layers, hidden_features, latent_features, out_channels):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.imgSize = imgSize
        self.n_hidden_layers = n_hidden_layers
        exec(f"self.l{n_hidden_layers+1} = nn.Linear(in_features = latent_features, out_features = hidden_features)")
        
        for i in range(n_hidden_layers,0,-1):
            exec(f"self.l{i} = nn.Linear(in_features = hidden_features, out_features = hidden_features)")
        self.l0 = nn.Linear(in_features = hidden_features, out_features = np.prod(imgSize)*out_channels)

    def forward(self, x):
        batch_size = x.shape[0]
        for i in range(self.n_hidden_layers+1,-1,-1):
            x = eval(f"self.l{i}(F.relu(x))")
        x = x.view(batch_size, self.out_channels, self.imgSize[0], self.imgSize[1])
        return x
    

class AEMnist2Hidden(BaseModel):
    def __init__(self, imgSize, n_hidden_layers, hidden_features, latent_features, cuda, in_channels):
        super(AEMnist2Hidden, self).__init__()
        self._cuda = cuda
        self.imgSize=imgSize
        self.latent_features = latent_features
        self.in_channels = in_channels
        self.hidden_features = hidden_features
        self.n_hidden_layers = n_hidden_layers
        
        
        # We encode the data onto the latent space using two linear layers
        self.encoder = Encoder(imgSize, n_hidden_layers, hidden_features, self.latent_features, in_channels=in_channels)
        # The latent code must be decoded into the original image
        self.decoder = Decoder(imgSize, n_hidden_layers, hidden_features, self.latent_features, out_channels=in_channels)
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
    
    