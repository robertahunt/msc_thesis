import torch
import torch.nn as nn
import torch.nn.functional as F

from .BaseModel import BaseModel


class MikkelAE(BaseModel):
    def __init__(self):
        super(MikkelAE, self).__init__()
        # We typically employ an "hourglass" structure
        # meaning that the decoder should be an encoder
        # in reverse.
        
        self.c1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=0)
        self.mp1 = nn.MaxPool2d(kernel_size=2,return_indices=True)
        self.c2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=0)
        self.c3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.mp2 = nn.MaxPool2d(kernel_size=2,return_indices=True)
        

        self.mup1 = nn.MaxUnpool2d(kernel_size=2)
        self.ct1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.ct2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=0)
        self.mup2 = nn.MaxUnpool2d(kernel_size=2)
        self.ct3 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=0)
        

    def forward(self, x): 
        outputs = {}
        # we don't apply an activation to the bottleneck layer
        #encoder
        x = self.c1(x)
        x = F.relu(x)
        size1 = x.size()
        x,ind1 = self.mp1(x)
        x = F.relu(x)
        x = self.c2(x)
        x = F.relu(x)
        x = self.c3(x)
        x = F.relu(x)
        size2 = x.size()
        x,ind2 = self.mp2(x)
        z = F.relu(x)
        
        #decoder
        x = self.mup1(z,ind2,output_size=size2)
        x = F.relu(x)
        x = self.ct1(x)
        x = F.relu(x)
        x = self.ct2(x)
        x = F.relu(x)
        x = self.mup2(x,ind1,output_size=size1)
        x = F.relu(x)
        x = self.ct3(x)
        
        
        # apply sigmoid to output to get pixel intensities between 0 and 1
        x_hat = torch.sigmoid(x)
        outputs['z'] = z
        outputs['x_hat'] = x_hat
        return outputs