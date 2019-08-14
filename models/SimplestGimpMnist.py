import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus

import numpy as np

from sklearn.manifold import TSNE
from .BaseModel import BaseModel
from .modelUtils import calc_size_after_mp, calc_size_after_conv, start_timer, tick, initialize_gmm, sample_from_gmm

    
class Encoder(BaseModel):
    def __init__(self, imgSize, latent_features, in_channels):
        super(Encoder, self).__init__()
        self.imgSize = imgSize
        self.in_channels = in_channels
        self.l1 = nn.Linear(in_features = np.prod(imgSize)*in_channels, out_features = latent_features)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, np.prod(self.imgSize)*self.in_channels)
        x = self.l1(x)
        return x, None
    
class Decoder(BaseModel):
    def __init__(self, imgSize, latent_features, num_samples, out_channels):
        super(Decoder, self).__init__()
        self.num_samples = num_samples
        self.l1 = nn.Linear(in_features = latent_features, out_features = np.prod(imgSize)*out_channels)

    def forward(self, x):
        x = self.l1(F.relu(x))
        return x
    

class SimplestGimpMnist(BaseModel):
    def __init__(self, imgSize, latent_features, num_samples, n_q_gaussians, cuda, in_channels, prior_initializer_name, prior_initializer_params, mc_points = 10000):
        super(SimplestGimpMnist, self).__init__()
        self._cuda = cuda
        self.imgSize=imgSize
        self.latent_features = latent_features
        self.in_channels = in_channels
        self.num_samples = num_samples
        self.n_p_gaussians = prior_initializer_params['n_gaussians']
        self.n_q_gaussians = n_q_gaussians
        self.mc_points = mc_points
        
        prior_initializer_params['d'] = self.latent_features
        prior_initializer_params['cuda'] = cuda
        self.p = initialize_gmm(prior_initializer_name, prior_initializer_params)

        # We encode the data onto the latent space using two linear layers
        self.encoder = Encoder(imgSize, (self.latent_features*2+1)*n_q_gaussians, in_channels=in_channels)
        # The latent code must be decoded into the original image
        self.decoder = Decoder(imgSize, self.latent_features, self.num_samples, out_channels=in_channels)
        self.device = torch.device("cuda:0" if cuda else "cpu")

        
    def plot_preprocess(self, outputs, batch, results, batchSize, cuda):
        num_samples = self.num_samples
        latent_variables = self.latent_features
        
        x, y, _id = batch
        y = np.array(y)
        
        x_hat = outputs['x_hat']
        decoder_z = outputs['decoder_z']
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
        z_plot = np.append(z_plot,p_mus.squeeze(2).squeeze(0).cpu().numpy(), axis=0)
        y_plot = np.append(y_plot,['p']*p_mus.shape[1])
        z_tsne = tsne.fit_transform(z_plot)
        for i in range(len(classes)):
            ax.scatter(*z_tsne[y_plot == classes[i]].T, c=colors[i], marker='o', label=classes[i], alpha=0.1)
        ax.scatter(*z_tsne[y_plot == 'p'].T, c='red', marker = 'x', label='p', s=30**2)
        
        ax.legend(np.append(classes,['p']))
        
    def take_samples(self, z, batch_size):
        mus_and_vars, ws = z[:,:-self.n_q_gaussians],z[:,-self.n_q_gaussians:]

        #split remaining z into mus and log_vars
        mus_and_vars = mus_and_vars.reshape(batch_size, self.n_q_gaussians, 1, self.latent_features*2)
        mus, log_vars = torch.chunk(mus_and_vars, 2, dim=-1)
        
                
        #ensure ws sum to 1
        ws = torch.softmax(ws, dim=1)
        
        # Make sure that the log variance is positive
        log_vars = softplus(log_vars)
        
        with torch.no_grad():
            epsilon = torch.randn(batch_size, 1, self.num_samples, self.latent_features) 
            if self._cuda:
                epsilon = epsilon.cuda()
        
        sigmas = torch.exp(log_vars/2)
        
        decoder_z = sample_from_gmm(self.num_samples, ws, mus, sigmas, self._cuda)
        
        # Run through decoder
        decoder_z = decoder_z.view(batch_size*self.num_samples, self.latent_features)
        return decoder_z, None, mus, sigmas, None, None, ws
        
    def forward(self, x): 
        batch_size = x.shape[0]
        outputs = {'x':x}
        
        # Split encoder outputs into a mean and variance vector
        z, _ = self.encoder(x) #z shape: [batch_size, 2*latent_features + n_gaussians]
        
        #split z into mus/vars and gaussian weights, w
        decoder_z, log_var, mus, sigmas, mu, _w, w = self.take_samples(z, batch_size)
        
       
        x = self.decoder(decoder_z)
        x = x.view(batch_size, self.num_samples, self.in_channels, self.imgSize[0], self.imgSize[1])
        # The original digits are on the scale [0, 1]
        x = torch.sigmoid(x)
        
        # Mean over samples
        x_hat = torch.mean(x, dim=1)       
        #print(mu.shape, mus.shape, sigmas.shape) 
        outputs["x_hat_all"] = x
        outputs["num_samples"] = self.num_samples
        outputs["x_hat"] = x_hat
        outputs["decoder_z"] = decoder_z
        outputs["p"] = self.p
        outputs["q"] = w, mus, sigmas
        outputs["cuda"] = self._cuda
        outputs["mc_points"] = self.mc_points
        
        return outputs
    
    