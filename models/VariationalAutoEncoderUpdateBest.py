import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus

import numpy as np

from .BaseModel import BaseModel
from .modelUtils import calc_size_after_mp, calc_size_after_conv
    
class Encoder(BaseModel):
    def __init__(self, imgSize, hidden_features, latent_features, in_channels):
        super(Encoder, self).__init__()
        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7)
        self.m2 = nn.MaxPool2d((2,2), stride=2)
        self.c3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.c4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.m5 = nn.MaxPool2d((2,2), stride=2)
        self.c6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.c7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.m8 = nn.MaxPool2d((2,2), stride=2)
        self.c9 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=8, stride=8)
        
        s = imgSize
        s1 = calc_size_after_conv(s, kernel=7)
        s2 = calc_size_after_mp(s1, kernel=2, stride=2)
        s3 = calc_size_after_conv(s2, kernel=3)
        s4 = calc_size_after_conv(s3, kernel=3)
        s5 = calc_size_after_mp(s4, kernel=2, stride=2)
        s6 = calc_size_after_conv(s5, kernel=3)
        s7 = calc_size_after_conv(s6, kernel=3)
        s8 = calc_size_after_mp(s7, kernel=2, stride=2)
        s9 = calc_size_after_conv(s8, kernel=8, stride=8)
        #s9 = tuple(np.array(s9) -1)
        self.linear_features = s9
        self.l10 = nn.Linear(in_features = np.prod(s9)*128, out_features = hidden_features)
        self.l11 = nn.Linear(in_features = hidden_features, out_features = latent_features)
        
    def forward(self, x):
        self.sizes = []
        x = self.c1(x)
        
        self.sizes += [x.size()]
        x = self.m2(F.relu(x))
        x = self.c3(F.relu(x))
        x = self.c4(F.relu(x))
        
        self.sizes += [x.size()]
        x = self.m5(F.relu(x))
        x = self.c6(F.relu(x))
        x = self.c7(F.relu(x))
        
        self.sizes += [x.size()]
        x = self.m8(F.relu(x))
        x = self.c9(F.relu(x))
        self.pre_flatten_size = x.size()
        x = x.view(x.size()[0],-1)
        x = self.l10(F.relu(x))
        x = self.l11(F.relu(x))
        return x, self.sizes
    
class Decoder(BaseModel):
    def __init__(self, pre_flatten_size, hidden_features, latent_features, num_samples, out_channels):
        super(Decoder, self).__init__()
        self.pre_flatten_size = pre_flatten_size
        self.num_samples = num_samples
        self.l11 = nn.Linear(in_features = latent_features, out_features = hidden_features)
        self.l10 = nn.Linear(in_features = hidden_features, out_features = np.prod(pre_flatten_size[2:])*128)
        
        self.c9 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=8, stride=8, output_padding=(0,2))
        self.m8 = nn.MaxUnpool2d((2,2), stride=2)
        self.c7 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3)
        self.c6 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3)
        self.m5 = nn.MaxUnpool2d((2,2), stride=2)
        self.c4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3)
        self.c3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3)
        self.m2 = nn.MaxUnpool2d((2,2), stride=2)
        self.c1 = nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=7)
        
    def forward(self, x, sizes, batch_size):
        x = self.l11(x)
        x = self.l10(F.relu(x))
        pre_flatten_size = torch.Size((x.size()[0],self.pre_flatten_size[1],self.pre_flatten_size[2],self.pre_flatten_size[3]))
        x = x.view(pre_flatten_size)
        x = self.c9(F.relu(x))
        
        x = F.interpolate(F.relu(x),size=sizes[2][2:])
        x = self.c7(F.relu(x))
        x = self.c6(F.relu(x))
        
        
        x = F.interpolate(F.relu(x),size=sizes[1][2:])
        x = self.c4(F.relu(x))
        x = self.c3(F.relu(x))
        
        
        x = F.interpolate(F.relu(x),size=sizes[0][2:])
        x = self.c1(F.relu(x))
        return x
    

class VariationalAutoEncoderUpdateBest(BaseModel):
    def __init__(self, imgSize, hidden_features, latent_features,num_samples, cuda, in_channels=3):
        super(VariationalAutoEncoderUpdateBest, self).__init__()
        self._cuda = cuda
        self.imgSize=imgSize
        self.hidden_features=hidden_features
        self.latent_features = latent_features
        self.num_samples=num_samples

        # We encode the data onto the latent space using two linear layers
        self.encoder = Encoder(imgSize, hidden_features, self.latent_features*2, in_channels=in_channels)#*2 to account for split in mean and variance
        self.pre_flatten_size = torch.Size([25, 128, 2, 3])
        # The latent code must be decoded into the original image
        self.decoder = Decoder(self.pre_flatten_size, hidden_features, self.latent_features, self.num_samples, out_channels=in_channels)
        

    def forward(self, x): 
        orig_shape = x.shape
        #x = x.view((x.shape[0],-1))
        outputs = {'x':x}
        
        # Split encoder outputs into a mean and variance vector
        z, mp_indices_sizes = self.encoder(x)
        mu, log_var = torch.chunk(z, 2, dim=-1)
        
        # Make sure that the log variance is positive
        log_var = softplus(log_var)
        
        # :- Reparametrisation trick
        # a sample from N(mu, sigma) is mu + sigma * epsilon
        # where epsilon ~ N(0, 1)
                
        # Don't propagate gradients through randomness
        with torch.no_grad():
            batch_size = mu.size(0)
            epsilon = torch.randn(batch_size, self.num_samples, self.latent_features)
            
            if self._cuda:
                epsilon = epsilon.cuda()
        
        sigma = torch.exp(log_var/2)
        
        # We will need to unsqueeze to turn
        # (batch_size, latent_dim) -> (batch_size, 1, latent_dim)
        
        z = mu.unsqueeze(1) + epsilon * sigma.unsqueeze(1)  
        
        a,b,c = z.size()
        # Run through decoder
        
        z = z.view(batch_size*self.num_samples, c)
        x = self.decoder(z, mp_indices_sizes, batch_size)
        a,b,c,d = x.size()
        x = x.view(batch_size, self.num_samples, b, c, d)
        # The original digits are on the scale [0, 1]
        x = torch.sigmoid(x)
        # Mean over samples
        x_hat = torch.mean(x, dim=1)
        #x_hat = x.view(orig_shape)
        
        outputs['epsilon'] = epsilon
        outputs["x_hat_all"] = x
        outputs["x_hat"] = x_hat
        outputs["z"] = mu
        outputs["mu"] = mu
        outputs["log_var"] = log_var
        
        return outputs