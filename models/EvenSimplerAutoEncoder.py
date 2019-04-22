import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus

import numpy as np

from .BaseModel import BaseModel
    
def calc_size_after_conv(s, kernel, stride=1, padding=0):
    h_in, w_in = s
    h_out = int((h_in+2*padding-kernel)/stride +1)
    w_out = int((w_in+2*padding-kernel)/stride +1)
    return (h_out, w_out)
    
def calc_size_after_mp(s, kernel, stride=1, padding=0, dilation=1):
    h_in, w_in = s
    h_out = int( ((h_in + 2 * padding - dilation * (kernel - 1)) / stride) + 1 )
    w_out = int( ((w_in + 2 * padding - dilation * (kernel - 1)) / stride) + 1 )
    return (h_out, w_out)
    
class Encoder(BaseModel):
    def __init__(self, imgSize, hidden_features, latent_features, in_channels):
        super(Encoder, self).__init__()
        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2)
        self.m2 = nn.MaxPool2d((2,2), stride=2)
        self.c3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1)
        self.c4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1)
        self.m5 = nn.MaxPool2d((2,2), stride=2)
        self.c6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=3)
        
        s = imgSize
        s1 = calc_size_after_conv(s, kernel=7, stride=2)
        s2 = calc_size_after_mp(s1, kernel=2, stride=2)
        s3 = calc_size_after_conv(s2, kernel=5, stride=1)
        s4 = calc_size_after_conv(s3, kernel=5, stride=1)
        s5 = calc_size_after_mp(s4, kernel=2, stride=2)
        s6 = calc_size_after_conv(s5, kernel=3, stride=3)
        self.linear_features = s6
      
        self.l10 = nn.Linear(in_features = np.prod(s6)*128, out_features = hidden_features)
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
        
        x = x.view(x.size()[0],-1)
        x = self.l10(F.relu(x))
        x = self.l11(F.relu(x))
        return x, self.sizes

    
class Decoder(BaseModel):
    def __init__(self, pre_flatten_size, hidden_features, latent_features, out_channels):
        super(Decoder, self).__init__()
        self.pre_flatten_size = pre_flatten_size
        self.l11 = nn.Linear(in_features = latent_features, out_features = hidden_features)
        self.l10 = nn.Linear(in_features = hidden_features, out_features = np.prod(pre_flatten_size[2:])*128)
        
        self.c6 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=3, output_padding=(0,1))

        self.c4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=1)
        self.c3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=1)

        self.c1 = nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=7, stride=2, output_padding=(1,1))
        
    def forward(self, x, sizes):
        x = self.l11(x)
        x = self.l10(F.relu(x))
        pre_flatten_size = torch.Size((x.size()[0],self.pre_flatten_size[1],self.pre_flatten_size[2],self.pre_flatten_size[3]))
        x = x.view(pre_flatten_size)
        x = self.c6(F.relu(x))
        x = F.interpolate(F.relu(x),size=sizes[1][2:])
        x = self.c4(F.relu(x))
        x = self.c3(F.relu(x))
        x = F.interpolate(F.relu(x),size=sizes[0][2:])
        x = self.c1(F.relu(x))
        return torch.sigmoid(x)
        
class EvenSimplerAutoEncoder(BaseModel):
    def __init__(self, imgSize, hidden_features, latent_features, in_channels=3):
        super(EvenSimplerAutoEncoder, self).__init__()
        self.encoder = Encoder(imgSize, hidden_features, latent_features, in_channels=in_channels)
        
        self.pre_flatten_size = torch.Size([25, 128, 1, 2])
        self.decoder = Decoder(self.pre_flatten_size, hidden_features, latent_features, out_channels=in_channels)
        
        
    def forward(self, x):
        outputs = dict()
        z, sizes = self.encoder(x)
        outputs['z'] = z
        x_hat = self.decoder(z, sizes)
        outputs['x'] = x
        outputs['x_hat'] = x_hat
        return outputs