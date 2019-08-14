import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus

import numpy as np

import sys
sys.path.append("..")
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
        self.c9 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.c10 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.m11 = nn.MaxPool2d((2,2), stride=2)
        
        s = imgSize
        s1 = calc_size_after_conv(s, kernel=7)
        s2 = calc_size_after_mp(s1, kernel=2, stride=2)
        s3 = calc_size_after_conv(s2, kernel=3)
        s4 = calc_size_after_conv(s3, kernel=3)
        s5 = calc_size_after_mp(s4, kernel=2, stride=2)
        s6 = calc_size_after_conv(s5, kernel=3)
        s7 = calc_size_after_conv(s6, kernel=3)
        s8 = calc_size_after_mp(s7, kernel=2, stride=2)
        s9 = calc_size_after_conv(s8, kernel=3)
        s10 = calc_size_after_conv(s9, kernel=3)
        s11 = calc_size_after_mp(s10, kernel=2, stride=2)
        #s9 = tuple(np.array(s9) -1)
        self.pre_flatten_size = s11
        self.l10 = nn.Linear(in_features = np.prod(s11)*128, out_features = hidden_features)
        self.l11 = nn.Linear(in_features = hidden_features, out_features = latent_features)
        
    def forward(self, x):
        self.sizes = []
        size = x.size()
        self.sizes += [size]
        
        x = self.c1(x)
        size = x.size()
        self.sizes += [size]
        x = self.m2(F.relu(x))
        
        x = self.c3(F.relu(x))
        x = self.c4(F.relu(x))
        size = x.size()
        self.sizes += [size]
        x = self.m5(F.relu(x))
        
        x = self.c6(F.relu(x))
        x = self.c7(F.relu(x))
        size = x.size()
        self.sizes += [size]
        x = self.m8(F.relu(x))
        
        x = self.c9(F.relu(x))
        x = self.c10(F.relu(x))
        size = x.size()
        self.sizes += [size]
        x = self.m11(F.relu(x))
        
        self.pre_flatten_size = x.size()
        x = x.view(x.size()[0],-1)
        x = self.l10(F.relu(x))
        x = self.l11(F.relu(x))
        return x, self.sizes
    
class Decoder(BaseModel):
    def __init__(self, pre_flatten_size, hidden_features, latent_features, out_channels):
        super(Decoder, self).__init__()
        self.pre_flatten_size = pre_flatten_size
        self.l11 = nn.Linear(in_features = latent_features, out_features = hidden_features)
        self.l10 = nn.Linear(in_features = hidden_features, out_features = np.prod(pre_flatten_size)*128)
        
        self.c10 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=3)
        self.c9 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=3)
        self.c7 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3)
        self.c6 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3)
        self.c4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3)
        self.c3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3)
        self.c1 = nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=7)
        
    def forward(self, x, sizes):
        x = self.l11(x)
        x = self.l10(F.relu(x))
        pre_flatten_shape = torch.Size((x.size()[0],sizes[4][1],self.pre_flatten_size[0],self.pre_flatten_size[1]))
        x = x.view(pre_flatten_shape)
        
        x = F.interpolate(F.relu(x),size=sizes[4][2:])
        x = self.c10(F.relu(x))
        x = self.c9(F.relu(x))
        
        x = F.interpolate(F.relu(x),size=sizes[3][2:])
        x = self.c7(F.relu(x))
        x = self.c6(F.relu(x))
        
        x = F.interpolate(F.relu(x),size=sizes[2][2:])
        x = self.c4(F.relu(x))
        x = self.c3(F.relu(x))
        
        x = F.interpolate(F.relu(x),size=sizes[1][2:])
        x = self.c1(F.relu(x))
        return torch.sigmoid(x)
        
class BasicAutoEncoderMoreLayers2(BaseModel):
    def __init__(self, imgSize, hidden_features, latent_features, in_channels=3):
        super(BasicAutoEncoderMoreLayers2, self).__init__()
        self.encoder = Encoder(imgSize, hidden_features, latent_features, in_channels=in_channels)
        self.decoder = Decoder(self.encoder.pre_flatten_size, hidden_features, latent_features, out_channels=in_channels)
        
        
    def forward(self, x):
        outputs = dict()
        z, indices = self.encoder(x)
        outputs['z'] = z
        x_hat = self.decoder(z, indices)
        outputs['x'] = x
        outputs['x_hat'] = x_hat
        return outputs