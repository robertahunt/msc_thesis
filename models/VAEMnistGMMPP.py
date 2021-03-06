import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus

import numpy as np

from sklearn.manifold import TSNE
from .BaseModel import BaseModel
from .modelUtils import calc_size_after_mp, calc_size_after_conv, make_random_GMM, start_timer, tick
    
class Encoder(BaseModel):
    def __init__(self, imgSize, hidden_features, latent_features, in_channels):
        super(Encoder, self).__init__()
        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.m3 = nn.MaxPool2d((2,2), stride=2)
        self.c4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.c5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.m6 = nn.MaxPool2d((2,2), stride=2)
        
        s = imgSize
        s1 = calc_size_after_conv(s, kernel=3, padding=1)
        s2 = calc_size_after_conv(s1, kernel=3, padding=1)
        s3 = calc_size_after_mp(s2, kernel=2, stride=2)
        s4 = calc_size_after_conv(s3, kernel=3, padding=1)
        s5 = calc_size_after_conv(s4, kernel=3, padding=1)
        s6 = calc_size_after_mp(s5, kernel=2, stride=2)
        #s9 = tuple(np.array(s9) -1)
        self.latent_features = s6
        
        self.l7 = nn.Linear(in_features = np.prod(s6)*128, out_features = hidden_features)
        self.l8 = nn.Linear(in_features = hidden_features, out_features = latent_features)
        
    def forward(self, x):
        self.sizes = []
        x = self.c1(x)
        x = self.c2(F.relu(x))
        
        self.sizes += [x.size()]
        x = self.m3(F.relu(x))
        x = self.c4(x)
        x = self.c5(F.relu(x))
        
        self.sizes += [x.size()]
        x = self.m6(F.relu(x))
        
        self.pre_flatten_size = x.size()
        
        x = x.view(x.size()[0],-1)
        x = self.l7(F.relu(x))
        x = self.l8(F.relu(x))
        return x, self.sizes
    
class Decoder(BaseModel):
    def __init__(self, pre_flatten_size, hidden_features, latent_features, num_samples, out_channels):
        super(Decoder, self).__init__()
        self.pre_flatten_size = pre_flatten_size
        self.num_samples = num_samples
        self.l8 = nn.Linear(in_features = latent_features, out_features = hidden_features)
        self.l7 = nn.Linear(in_features = hidden_features, out_features = np.prod(pre_flatten_size[2:])*128)
        
        self.c5 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.c4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.c2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.c1 = nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=3, padding=1)
        
    def forward(self, x, sizes, batch_size):
        x = self.l8(x)
        x = self.l7(F.relu(x))
        pre_flatten_size = torch.Size((x.size()[0],self.pre_flatten_size[1],self.pre_flatten_size[2],self.pre_flatten_size[3]))
        x = x.view(pre_flatten_size)
        
        x = F.interpolate(x,size=sizes[1][2:])
        x = self.c5(F.relu(x))
        x = self.c4(F.relu(x))
        
        x = F.interpolate(x,size=sizes[0][2:])
        x = self.c2(F.relu(x))
        x = self.c1(F.relu(x))
        return x
    

class VAEMnistGMMPP(BaseModel):
    def __init__(self, imgSize, hidden_features, latent_features, num_samples, cuda, in_channels, n_p_gaussians, n_q_gaussians, min_mu, max_mu, sigma, mc_points = 10000):
        super(VAEMnistGMMPP, self).__init__()
        self._cuda = cuda
        self.imgSize=imgSize
        self.hidden_features=hidden_features
        self.latent_features = latent_features
        self.in_channels = in_channels
        self.num_samples=num_samples
        self.n_p_gaussians=n_p_gaussians
        self.n_q_gaussians=n_q_gaussians
        self.p = make_random_GMM(latent_features, n_p_gaussians, sigma, min_mu, max_mu, cuda)
        self.mc_points = mc_points
        

        # We encode the data onto the latent space using two linear layers
        self.encoder = Encoder(imgSize, hidden_features, (self.latent_features*2+1)*n_q_gaussians, in_channels=in_channels)#*2 to account for split in mean and variance
        self.pre_flatten_size = torch.Size([25, 128, 2, 3])
        # The latent code must be decoded into the original image
        self.decoder = Decoder(self.pre_flatten_size, hidden_features, self.latent_features, self.num_samples, out_channels=in_channels)
        self.device = torch.device("cuda:0" if cuda else "cpu")

        
    def plot_preprocess(self, outputs, batch, results, batchSize, cuda):
        num_samples = self.num_samples
        latent_variables = self.latent_features
        
        x, y, _id = batch
        y = np.array(y)
        
        x_hat = outputs['x_hat']
        decoder_z = outputs['decoder_z'].view(outputs['mu'].shape)
        decoder_z = decoder_z.cpu().detach().numpy().reshape(-1, latent_variables)
        y = y.reshape((batchSize, 1, 1, 1))
        y = np.repeat(y, num_samples, axis=1)
        y = y.squeeze().flatten()
        

        if results is not None:
            z_train, y_train, z_valid, y_valid = results
            z_plot = z_valid
            y_plot = np.array(y_valid)
        else:
            z_plot = decoder_z#z.cpu().detach().numpy()
            y_plot = y

        classes = np.unique(y_plot)
        if cuda:
            x = x.cpu().detach()
            x_hat = x_hat.cpu().detach()
            
        return x, x_hat, z_plot, y_plot, classes
    
    def _plot_tsne(self, ax, z_plot, y_plot, classes, colors):
        ax.set_title('Latent space - TSNE')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        tsne = TSNE(n_components=2, perplexity=20)
        p_w, p_mus, p_sigmas = self.p
        z_plot = np.append(z_plot,p_mus.squeeze(3).squeeze(0).cpu().numpy(), axis=0)
        y_plot = np.append(y_plot,['p']*p_mus.shape[1])
        z_tsne = tsne.fit_transform(z_plot)
        for i in range(len(classes)):
            ax.scatter(*z_tsne[y_plot == classes[i]].T, c=colors[i], marker='o', label=classes[i], alpha=0.1)
        ax.scatter(*z_tsne[y_plot == 'p'].T, c='red', marker = 'x', label='p', s=30**2)
        
        ax.legend(np.append(classes,['p']))
        
    def take_samples(self, z, batch_size):
        mus_and_vars, w = z[:,:-self.n_q_gaussians],z[:,-self.n_q_gaussians:]

        #split remaining z into mus and log_vars
        mus_and_vars = mus_and_vars.reshape(batch_size, 1, self.n_q_gaussians, self.latent_features*2)
        mus, log_vars = torch.chunk(mus_and_vars, 2, dim=-1)
        
                
        #ensure ws sum to 1
        w = torch.softmax(w, dim=1)
        
        
        # Make sure that the log variance is positive
        log_vars = softplus(log_vars)
        
        with torch.no_grad():
            epsilon = torch.randn(batch_size, self.num_samples, 1, self.latent_features) 
            if self._cuda:
                epsilon = epsilon.cuda()
        
        sigmas = torch.exp(log_vars/2)
        #first in mixture model, figure out which gaussian each is a part of
        _w = torch.multinomial(w, num_samples=self.num_samples, replacement=True).unsqueeze(2).unsqueeze(3)
        #get mu and sigma for that gaussian
        mu = torch.gather(mus.repeat(1,self.num_samples,1,1), 2, _w.repeat(1,1,1,self.latent_features)) #mu shape: [batch_size, 1, latent_features]
        sigma = torch.gather(sigmas.repeat(1,self.num_samples,1,1), 2, _w.repeat(1,1,1,self.latent_features)) #sigma shape: [batch_size, 1, latent_features]
        
        decoder_z = mu + epsilon * sigma
        log_var = 2*torch.log(sigma)
       
        
        
        # Run through decoder
        decoder_z = decoder_z.view(batch_size*self.num_samples, self.latent_features)
        return decoder_z, log_var, mus, sigmas, mu, _w, w
        
    def forward(self, x): 
        batch_size = x.shape[0]
        outputs = {'x':x}
        
        # Split encoder outputs into a mean and variance vector
        z, encoder_sizes = self.encoder(x) #z shape: [batch_size, 2*latent_features + n_gaussians]
        #split z into mus/vars and gaussian weights, w
        decoder_z, log_var, mus, sigmas, mu, _w, w = self.take_samples(z, batch_size)
        
       
        x = self.decoder(decoder_z, encoder_sizes, batch_size)
        x = x.view(batch_size, self.num_samples, self.in_channels, self.imgSize[0], self.imgSize[1])
        # The original digits are on the scale [0, 1]
        x = torch.sigmoid(x)
        
        # Mean over samples
        x_hat = torch.mean(x, dim=1)       
        #print(mu.shape, mus.shape, sigmas.shape) 
        outputs["x_hat_all"] = x
        outputs["x_hat"] = x_hat
        outputs["decoder_z"] = decoder_z
        outputs["chosen_w"] = _w
        outputs["mu"] = mu
        outputs["log_var"] = log_var.squeeze(dim=1)
        outputs["p"] = self.p
        outputs["q"] = w, mus.squeeze(dim=1).unsqueeze(3), sigmas.squeeze(dim=1).unsqueeze(3)
        outputs["cuda"] = self._cuda
        outputs["mc_points"] = self.mc_points
        
        return outputs
    
    