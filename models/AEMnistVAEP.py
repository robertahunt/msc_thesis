import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus

import numpy as np

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from .BaseModel import BaseModel
from .modelUtils import calc_size_after_mp, calc_size_after_conv, make_random_GMM, start_timer, tick, initialize_gmm, initialize_zero_mean_gauss

from bokeh.plotting import figure
    
class Encoder(BaseModel):
    def __init__(self, imgSize, n_conv_layers, filter_size, n_filters, in_channels, latent_features, stride=1):
        super(Encoder, self).__init__()
        self.imgSize = imgSize
        self.in_channels = in_channels
        self.n_conv_layers = n_conv_layers
        self.filter_size = filter_size
        self.padding = filter_size//2
        self.latent_features = latent_features
        self.stride = stride
        self.n_filters = n_filters
        self.c0 = nn.Conv2d(in_channels=in_channels, out_channels=n_filters, kernel_size=filter_size, padding=self.padding, stride=stride)
        self.sizes = [calc_size_after_conv(imgSize, filter_size, stride=stride, padding=self.padding)]
        for i in range(1,n_conv_layers):
            exec(f"self.c{i} = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=filter_size, padding=self.padding, stride=stride)")      
            self.sizes += [calc_size_after_conv(self.sizes[-1], filter_size, stride=stride, padding=self.padding)]
        
        self.l0 = nn.Linear(in_features = np.prod(self.sizes[-1])*n_filters, out_features = latent_features)
        self.l1 = nn.Linear(in_features = latent_features, out_features = latent_features)
        self.l2 = nn.Linear(in_features = latent_features, out_features = latent_features)
        
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
    def __init__(self, imgSize, n_conv_layers, filter_size, n_filters, num_samples, out_channels, latent_features, sizes, stride=1):
        super(Decoder, self).__init__()
        self.num_samples = num_samples
        self.out_channels = out_channels
        self.imgSize = imgSize
        self.n_conv_layers = n_conv_layers
        self.filter_size = filter_size
        self.padding = filter_size//2
        self.stride = stride
        self.n_filters = n_filters
        self.sizes = sizes
        self.latent_features = latent_features
        
        self.l2 = nn.Linear(in_features = latent_features, out_features = latent_features)
        self.l1 = nn.Linear(in_features = latent_features, out_features = latent_features)
        self.l0 = nn.Linear(in_features = latent_features, out_features = np.prod(sizes[-1])*n_filters)

        
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
    
    
    

class AEMnistVAEP(BaseModel):
    def __init__(self, imgSize, n_conv_layers, filter_size, n_filters, stride, in_channels, n_q_gaussians, cuda, latent_features, num_samples, prior_initializer_name=None, prior_initializer_params=None, mc_points = 10000):   
        super(AEMnistVAEP, self).__init__()
        self._cuda = cuda
        self.imgSize = imgSize
        self.n_conv_layers = n_conv_layers
        self.filter_size = filter_size
        self.n_filters = n_filters
        self.stride = stride
        self.latent_features = latent_features
        self.in_channels = in_channels
        self.num_samples = num_samples
        self.n_p_gaussians = prior_initializer_params['n_gaussians'] if prior_initializer_params is not None else 1
        self.n_q_gaussians = n_q_gaussians
        self.mc_points = mc_points
        
        if prior_initializer_name is None:
            self.p = initialize_zero_mean_gauss(sigma=1, d=self.latent_features, cuda=cuda)
        else:
            prior_initializer_params['d'] = self.latent_features
            prior_initializer_params['cuda'] = cuda
            self.p = initialize_gmm(prior_initializer_name, prior_initializer_params)

        # We encode the data onto the latent space using two linear layers
        self.encoder = Encoder(imgSize, n_conv_layers, filter_size, n_filters, in_channels, (self.latent_features*2+1)*n_q_gaussians, stride=stride)
        
        # The latent code must be decoded into the original image
        self.decoder = Decoder(imgSize, n_conv_layers, filter_size, n_filters, num_samples, in_channels, latent_features, sizes = self.encoder.sizes, stride=stride)
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
    
    def _plot_tsne(self, z_plot, y_plot, classes, colors):
        tsne = TSNE(n_components=2, perplexity=20)
        z_tsne = tsne.fit_transform(z_plot)
        p_w, p_mus, p_sigmas = self.p
        z_plot = np.append(z_plot,p_mus.squeeze(2).squeeze(0).cpu().numpy(), axis=0)
        y_plot = np.append(y_plot,['p']*p_mus.shape[1])
        z_tsne = tsne.fit_transform(z_plot)
        
        p_tsne = figure(width=300, plot_height=300, title="Latent Space - TSNE")
        
        for i in range(len(classes)):
            p_tsne.scatter(*z_tsne[y_plot == classes[i]].T, color=colors[i], marker='o', legend=classes[i], alpha=0.1)
        p_tsne.scatter(*z_tsne[y_plot == 'p'].T, color='red', marker = 'x', legend='p', size=30)
        p_tsne.xaxis.axis_label = 'Dimension 1'
        p_tsne.yaxis.axis_label = 'Dimension 2'
        return p_tsne
        
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
        
    def reset_prior(self, z):
        print(cdist(self.p[1].squeeze().cpu().detach().numpy(),self.p[1].squeeze().cpu().detach().numpy()))
        clust = KMeans(n_clusters=self.n_p_gaussians)
        clust.fit(z.cpu().detach().numpy())
        new_mus = torch.Tensor(clust.cluster_centers_).unsqueeze(0).unsqueeze(2)
        print(cdist(clust.cluster_centers_,clust.cluster_centers_))
        if self.cuda:
            return (self.p[0], new_mus.cuda(), self.p[2])
        else:
            return (self.p[0], new_mus, self.p[2])
        
        
    def forward(self, x, reset_prior=False): 
        batch_size = x.shape[0]
        outputs = {'x':x}
        z, _ = self.encoder(x)
        decoder_z, log_var, mus, sigmas, mu, _w, w = self.take_samples(z, batch_size)
        b = torch.exp(log_var/2)/2
        
        if reset_prior==True:
            self.p = self.reset_prior(decoder_z)
       
        x = self.decoder(decoder_z)
        x = x.view(batch_size, self.num_samples, self.in_channels, self.imgSize[0], self.imgSize[1])
        # The original digits are on the scale [0, 1]
        x = torch.sigmoid(x)
        
        # Mean over samples
        x_hat = torch.mean(x, dim=1)       
        #print(mu.shape, mus.shape, sigmas.shape) 
        outputs["x_hat_all"] = x
        outputs["x_hat"] = x_hat
        outputs['num_samples'] = self.num_samples
        outputs["decoder_z"] = decoder_z
        outputs["chosen_w"] = _w
        outputs["mu"] = mu
        outputs["log_var"] = log_var.squeeze(dim=1)
        outputs["p"] = self.p
        outputs["q"] = w, mus.squeeze(dim=1).unsqueeze(2), sigmas.squeeze(dim=1).unsqueeze(2)
        outputs['b'] = b
        outputs["cuda"] = self._cuda
        outputs["mc_points"] = self.mc_points
        
        return outputs
    
    