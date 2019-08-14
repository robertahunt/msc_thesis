import torch
import numpy as np
from scipy.spatial.distance import cdist
import os
import cv2
import sys

import numpy as np
import pandas as pd


global start_time
def start_timer():
    global start_time
    start_time = pd.Timestamp.now()
    
def tick(msg=''):
    global start_time
    print(msg + ', Time Taken: %s'%str(pd.Timestamp.now() - start_time))
    
def calc_size_after_conv(s, kernel, stride=1, padding=0):
    h_in, w_in = s
    h_out = int((h_in+2*padding-kernel)/stride +1)
    w_out = int((w_in+2*padding-kernel)/stride +1)
    return (h_out, w_out)
    
def calc_size_after_mp(s, kernel, stride=1, padding=0, dilation=1):
    h_in, w_in = s
    h_out = int( ((h_in + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1 )
    w_out = int( ((w_in + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1 )
    return (h_out, w_out)

#def make_GMM(d, n_gaussians, mu_min, mu_max, std_min, std_max, cuda):
#    w = torch.rand(n_gaussians)
#    w = w/torch.sum(w)
#    mus = torch.rand(n_gaussians,d)*(mu_max - mu_min) + mu_min
#    covs = torch.rand(n_gaussians,d)*(std_max - std_min) + std_min
#    
#    if cuda:
#        w = w.cuda()
#        mus = mus.cuda()
#        covs = covs.cuda()
#    return w.unsqueeze(0), mus.unsqueeze(0).unsqueeze(3), covs.unsqueeze(0).unsqueeze(3)

def make_random_GMM(d, n_gaussians, sigma, mu_min, mu_max, cuda):
    max_loops = 100

    #make sure it is likely that a solution can be found
    #assert (mu_max - mu_min)*d*3/(6*sigma) > n_gaussians

    means = [np.random.rand(d)*(mu_max - mu_min) + mu_min]
    for i in range(n_gaussians - 1):
        candidate_mean = [np.random.rand(d)*(mu_max - mu_min) + mu_min]
        min_dist = cdist(means,candidate_mean).min()
        l = 0
        while (min_dist < sigma*6) & (l < max_loops):
            candidate_mean = [np.random.rand(d)*(mu_max - mu_min) + mu_min]
            min_dist = cdist(means,candidate_mean).min()
            l += 1
        if l == max_loops:
            print('Max loops reached when building GT gaussian, ending.')
            assert False
        means += candidate_mean

    with torch.no_grad():
        w = torch.Tensor([1/n_gaussians]*n_gaussians).unsqueeze(0)
        mus = torch.Tensor(means).unsqueeze(0).unsqueeze(2)
        covs = torch.Tensor(torch.ones((mus.shape))*sigma)

        if cuda:
            w = w.cuda()
            mus = mus.cuda()
            covs = covs.cuda()
    return w, mus, covs

def make_orthogonal_GMM(d, n_gaussians, sigma, cuda):
    #Place each gaussian 6 sigmas along each axis, so all 'orthogonal' to eachother
    
    #make sure it is possible
    assert d >= n_gaussians

    means = []
    for i in range(n_gaussians):
        mean = np.zeros(d)
        mean[i] = sigma*6
        means += [mean]

    with torch.no_grad():
        w = torch.Tensor([1/n_gaussians]*n_gaussians).unsqueeze(0)
        mus = torch.Tensor(means).unsqueeze(0).unsqueeze(2)
        covs = torch.Tensor(torch.ones((mus.shape))*sigma)
        print(w.shape, mus.shape, covs.shape)
        if cuda:
            w = w.cuda()
            mus = mus.cuda()
            covs = covs.cuda()
    return w, mus, covs

def initialize_gmm(init_name, init_params):
    assert init_name in gmm_initializer_dict.keys(), f"GMM initializer {init_name} not found, valid options: {gmm_initializer_dict.keys()}"
    initializer = gmm_initializer_dict[init_name]
    gmm = initializer(**init_params)
    return gmm

def initialize_zero_mean_gauss(sigma, d, cuda):
    means = [np.zeros(d)]
    w = torch.Tensor([1]).unsqueeze(0)
    
    means = torch.Tensor(means).unsqueeze(0).unsqueeze(2)
    sigmas = torch.Tensor(torch.ones((means.shape))*sigma)
    if cuda:
        w = w.cuda()
        means = means.cuda()
        sigmas = sigmas.cuda()
        
    return w, means, sigmas

gmm_initializer_dict = {
    'random': make_random_GMM,
    'orthogonal': make_orthogonal_GMM,
}

def sample_from_gmm_backup(n_samples, ws, mus, sigmas, cuda):
    batch_size = ws.shape[0]
    d = mus.shape[-2]
    
    with torch.no_grad():
        chosen_gaussians = torch.multinomial(ws,n_samples, replacement=True).view(batch_size, 1, 1, n_samples).repeat(1,1,d,1)
        ep = torch.randn(batch_size, 1, d, n_samples)
        
        if cuda:
            chosen_gaussians = chosen_gaussians.cuda()
            ep = ep.cuda()
    
    #sample n_points from the f function, these samples are k_i
    mu = torch.gather(mus.repeat(1,1,1,n_samples), 1, chosen_gaussians)
    sigma = torch.gather(sigmas.repeat(1,1,1,n_samples), 1, chosen_gaussians)
    u_i = mu + ep*sigma
    return u_i

def sample_from_gmm(n_samples, ws, mus, sigmas, cuda):
    batch_size = ws.shape[0]
    d = mus.shape[-1]
    
    with torch.no_grad():
        chosen_gaussians = torch.multinomial(ws,n_samples, replacement=True).view(batch_size, 1,  n_samples, 1).repeat(1,1,1,d)
        ep = torch.randn(batch_size, 1, n_samples, d)
        
        if cuda:
            chosen_gaussians = chosen_gaussians.cuda()
            ep = ep.cuda()
    
    #sample n_points from the f function, these samples are k_i
    mu = torch.gather(mus.repeat(1,1, n_samples,1), 1, chosen_gaussians)
    sigma = torch.gather(sigmas.repeat(1,1,n_samples,1), 1, chosen_gaussians)
    u_i = mu + ep*sigma
    return u_i
    