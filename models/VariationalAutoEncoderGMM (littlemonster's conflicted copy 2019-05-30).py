import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus

import numpy as np

from .BaseModel import BaseModel
from .modelUtils import calc_size_after_mp, calc_size_after_conv, make_GMM
    
class Encoder(BaseModel):
    def __init__(self, imgSize, hidden_features, latent_features, in_channels, n_gaussians):
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
    def __init__(self, pre_flatten_size, hidden_features, latent_features, num_samples, out_channels, n_gaussians):
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
    

class VariationalAutoEncoderGMM(BaseModel):
    def __init__(self, imgSize, hidden_features, latent_features, num_samples, cuda, in_channels, n_gaussians, min_mu, max_mu, sigma):
        super(VariationalAutoEncoderGMM, self).__init__()
        self._cuda = cuda
        self.imgSize=imgSize
        self.hidden_features=hidden_features
        self.latent_features = latent_features
        self.in_channels = in_channels
        self.num_samples=num_samples
        self.n_gaussians=n_gaussians
        self.g = make_GMM(latent_features, n_gaussians, sigma, min_mu, max_mu, cuda)

        # We encode the data onto the latent space using two linear layers
        self.encoder = Encoder(imgSize, hidden_features, (self.latent_features*2+1)*n_gaussians, in_channels=in_channels, n_gaussians=n_gaussians)#*2 to account for split in mean and variance
        self.pre_flatten_size = torch.Size([25, 128, 2, 3])
        # The latent code must be decoded into the original image
        self.decoder = Decoder(self.pre_flatten_size, hidden_features, self.latent_features, self.num_samples, out_channels=in_channels, n_gaussians=n_gaussians)
        self.device = torch.device("cuda:0" if cuda else "cpu")

    def forward(self, x): 
        batch_size = x.shape[0]
        outputs = {'x':x}
        
        # Split encoder outputs into a mean and variance vector
        z, encoder_sizes = self.encoder(x) #z shape: [batch_size, 2*latent_features + n_gaussians]
        
        #split z into mus/vars and gaussian weights, w
        mus_and_vars, w = z[:,:-self.n_gaussians],z[:,-self.n_gaussians:]

        #split remaining z into mus and log_vars
        mus_and_vars = mus_and_vars.reshape(batch_size, 1, self.n_gaussians, self.latent_features*2)
        mus, log_vars = torch.chunk(mus_and_vars, 2, dim=-1)
                
        #ensure ws sum to 1
        w = torch.softmax(w, dim=1)
        
        # Make sure that the log variance is positive
        log_vars = softplus(log_vars)
        
        with torch.no_grad():
            epsilon = torch.randn(batch_size, self.num_samples, self.latent_features) 
            
            if self._cuda:
                epsilon = epsilon.cuda()
        
        sigmas = torch.exp(log_vars/2)
                
        #first in mixture model, figure out which gaussian each is a part of
        #shouldn't there be 15 samples for each?
        _w = torch.multinomial(w, num_samples=1).unsqueeze(1).unsqueeze(3).repeat(1,1,1,self.latent_features)
        
        #get mu and sigma for that gaussian
        mu = torch.gather(mus, 2, _w).squeeze(1) #mu shape: [batch_size, 1, latent_features]
        sigma = torch.gather(sigmas, 2, _w).squeeze(1) #sigma shape: [batch_size, 1, latent_features]
        
        #_w = (w == w.max(dim=1)[0].unsqueeze(1)).unsqueeze(1).unsqueeze(3)
        #_w = _w.to(self.device, dtype=torch.float32)
        #w now contains a tensor with 1s and 0s.
        #mu = (_w*mus).sum(dim=2)
        #sigma = (_w*sigmas).sum(dim=2)  
        
        decoder_z = mu + epsilon * sigma
        log_var = 2*torch.log(sigma)
        
        
        # Run through decoder
        decoder_z = decoder_z.view(batch_size*self.num_samples, self.latent_features)
        x = self.decoder(decoder_z, encoder_sizes, batch_size)
        x = x.view(batch_size, self.num_samples, self.in_channels, self.imgSize[0], self.imgSize[1])
        # The original digits are on the scale [0, 1]
        x = torch.sigmoid(x)
        
        # Mean over samples
        x_hat = torch.mean(x, dim=1)        
        
        #print(mu.shape, mus.shape, sigmas.shape) 
        outputs['epsilon'] = epsilon
        #outputs["x_hat_all"] = x
        outputs["x_hat"] = x_hat
        outputs["z"] = mu.squeeze(dim=1)
        outputs["w"] = w
        outputs["mus"] = mus.squeeze(dim=1)
        outputs["mu"] = mu
        outputs["sigmas"] = sigmas.squeeze(dim=1)
        outputs["log_var"] = log_var.squeeze(dim=1)
        outputs["g"] = self.g
        
        return outputs