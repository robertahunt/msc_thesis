import os
import torch
import numpy as np
import torch.optim as optim
import torchvision.transforms
import torch.nn.functional as F

from glob import glob
from scipy.stats import multivariate_normal
from torch.nn.functional import binary_cross_entropy

import myTransforms
import datagetters
from models.modelUtils import sample_from_gmm

def get_model_called(model_name, params):
    model_fps = glob('/home/rob/Dropbox/thesis/2. code/src/models/*.py')
    model_names = [os.path.basename(fp).split('.')[0] for fp in model_fps]
    model_names.remove('__init__')
    if model_name == '':
        print(model_names)
        return False
    if model_name in model_names:
        exec(f'from models.{model_name} import {model_name}',globals())
        exec(f'model = {model_name}(**{params})',globals())
        return model
    else:
        return False

def get_loss_function_called(loss_name):
    assert loss_name in loss_function_dict.keys(), f"loss function {loss_name} not found, valid options: {loss_function_dict.keys()}"
    return loss_function_dict[loss_name]



def get_optimizer_called(opt_name, netparams, hparams):
    assert opt_name in optimizer_dict.keys(), f"optimizer {opt_name} not found, valid options: {optimizer_dict.keys()}"
    return optimizer_dict[opt_name](netparams, **hparams)

def get_transform_called(tr_name, params=dict()):
    assert tr_name in transforms_dict.keys(), f"transform '{tr_name}' not found, valid options: {transforms_dict.keys()}"
    return transforms_dict[tr_name](**params)

def get_datagetter_called(datagetter_name, params=dict()):
    assert datagetter_name in datagetter_dict.keys(), f"datagetter '{datagetter_name}' not found, valid options: {datagetter_dict.keys()}"
    return datagetter_dict[datagetter_name](**params)

def mse_loss(outputs):
    loss = F.mse_loss(outputs['x'],outputs['x_hat'])
    return loss, [loss]

def bce_loss(outputs):
    loss = F.binary_cross_entropy(outputs['x_hat'],outputs['x'],reduction="none").mean()
    return loss, [loss]



def get_pdf(k, ws, mus, sigmas, cuda):
    #print(k.shape, ws.shape, mus.shape, sigmas.shape)
    #torch.Size([7, n_g, d, n_points]) torch.Size([7, n_g]) torch.Size([7, n_g, d, n_points]) torch.Size([7, n_g, d, n_points])
    
    #torch.Size([7, 1, 100, 1000]) torch.Size([1, 2]) torch.Size([1, 2, 100, 1]) torch.Size([1, 2, 100, 1])
    #print(mus[:,:,0,0])
    #a,b,c,d = mus.shape
    #print(mus.view(1,a*b,c,d)[:,:,0,0])
    #get the probability density function
    #at each of the n_points points
    #for a mixed gaussian model with weights given by ws,
    #means given by mus and diagonal covariances given by sigmas

    #this is the exponent of the multivariate gaussian
    #The equation for a multivariate gaussian is :
    #  (2*pi)^(d/2)*det(cov)^(-1/2)*e^((1/2)*(x-mu).T*inverse_cov*(x-mu))
    #in our case the calculation is simplified because
    #we assume the covariance matrix is diagonal, so the determinant is just the product
    #of the elements, and the inverse is just 1/sigma
    latent_variables = mus.shape[2]
    ws = ws.unsqueeze(2).unsqueeze(3)
    covs = sigmas**2

    #to avoid infinities - use log covariance
    log_covs = torch.log(covs)

    log_det_cov = (torch.sum(log_covs, 2)).unsqueeze(2)
    exponent = (-1/2)*((k-mus)*(1/covs)*(k-mus)).sum(dim=2, keepdim=True)

    if cuda:
        ws = ws.type(torch.cuda.DoubleTensor)
        log_det_cov = log_det_cov.type(torch.cuda.DoubleTensor)
        exponent = exponent.type(torch.cuda.DoubleTensor)
    else:
        ws = ws.type(torch.DoubleTensor)
        log_det_cov = log_det_cov.type(torch.DoubleTensor)
        exponent = exponent.type(torch.DoubleTensor)

    #multiplier = ((2*np.pi)**(-latent_variables/2))*np.e**((-1/2)*log_det_cov)
    #ignore first part of multiplier because it cancels out in mc approximation anyways
    multiplier = np.e**((-1/2)*log_det_cov)

    if cuda:
        multiplier = multiplier.type(torch.cuda.DoubleTensor)
    else:
        multiplier = multiplier.type(torch.DoubleTensor)
    pdf = multiplier*np.e**exponent
    #since ours is a mixture of gaussians, we sum the pdf over the number of gaussians we have 
    #multiplied by the weight each gaussian has
    pdf = ((ws*pdf).sum(dim=1,keepdim=True))#.type(torch.cuda.DoubleTensor)   
    return pdf

def get_pdf_all(k, ws, mus, sigmas, cuda):
    #Get the pdf of the entire batch at once.
    batch_size_k = k.shape[0]
    batch_size_g = mus.shape[0]
    n_g = mus.shape[1]
    n_samples = k.shape[2]
    d = mus.shape[3]
    
    #reshape k, so batch size is incorporated into just more samples
    k = k.reshape(1, 1, batch_size_k*n_samples, d)
    
    #change ws shape to [1,batch_size*n_gaussians], so treat as if it were a GMM of the entire batch
    ws = ws.reshape(1,batch_size_g*n_g)
    ws = ws/ws.sum(dim=1,keepdim=True)
    
    mus = mus.reshape(1,batch_size_g*n_g,1,d)
    sigmas = sigmas.reshape(1,batch_size_g*n_g,1,d)
    
    #get the probability density function
    #at each of the n_points points
    #for a mixed gaussian model with weights given by ws,
    #means given by mus and diagonal covariances given by sigmas

    #this is the exponent of the multivariate gaussian
    #The equation for a multivariate gaussian is :
    #  (2*pi)^(d/2)*det(cov)^(-1/2)*e^((1/2)*(x-mu).T*inverse_cov*(x-mu))
    #in our case the calculation is simplified because
    #we assume the covariance matrix is diagonal, so the determinant is just the product
    #of the elements, and the inverse is just 1/sigma
    ws = ws.unsqueeze(2).unsqueeze(3)
    covs = sigmas**2

    #to avoid infinities - use log covariance
    log_covs = torch.log(covs)
    log_det_cov = (torch.sum(log_covs, 3)).unsqueeze(3)
    exponent = (-1/2)*((k-mus)*(1/covs)*(k-mus)).sum(dim=3, keepdim=True)

    ws = ws.double()
    log_det_cov = log_det_cov.double()
    exponent = exponent.double()

    #multiplier = ((2*np.pi)**(-latent_variables/2))*np.e**((-1/2)*log_det_cov)
    #ignore first part of multiplier because it cancels out in mc approximation anyways
    
    multiplier = np.e**((-1/2)*log_det_cov)
    multiplier = multiplier.double()

    pdf = multiplier*np.e**exponent
    #since ours is a mixture of gaussians, we sum the pdf over the number of gaussians we have 
    #multiplied by the weight each gaussian has
    pdf = ((ws*pdf).sum(dim=1,keepdim=True))
    return pdf

def get_pdf_all_weird_prior(k, p_ws, q_mus, p_sigmas, cuda):
    #Get the pdf of the entire batch at once.
    batch_size_k = k.shape[0]
    batch_size_g = p_ws.shape[0]
    n_g = p_ws.shape[1]
    n_samples = k.shape[2]
    d = q_mus.shape[3]
    batch_size_q = q_mus.shape[0]
    n_g_q = q_mus.shape[1]
    
    #reshape k, so batch size is incorporated into just more samples
    k = k.reshape(1, 1, batch_size_k*n_samples, d)
    p_ws = p_ws.repeat(batch_size_q,1)
    p_sigmas = p_sigmas.repeat(batch_size_q,1,1,1)
    
    #change ws shape to [1,batch_size*n_gaussians], so treat as if it were a GMM of the entire batch
    p_ws = p_ws.reshape(1,batch_size_q*n_g)
    p_ws = p_ws/p_ws.sum(dim=1,keepdim=True)
    
    q_mus = q_mus.reshape(1,batch_size_q*n_g_q,1,d)
    p_sigmas = p_sigmas.reshape(1,batch_size_q*n_g,1,d)
    
    #get the probability density function
    #at each of the n_points points
    #for a mixed gaussian model with weights given by ws,
    #means given by mus and diagonal covariances given by sigmas

    #this is the exponent of the multivariate gaussian
    #The equation for a multivariate gaussian is :
    #  (2*pi)^(d/2)*det(cov)^(-1/2)*e^((1/2)*(x-mu).T*inverse_cov*(x-mu))
    #in our case the calculation is simplified because
    #we assume the covariance matrix is diagonal, so the determinant is just the product
    #of the elements, and the inverse is just 1/sigma
    p_ws = p_ws.unsqueeze(2).unsqueeze(3)
    p_covs = p_sigmas**2

    #to avoid infinities - use log covariance
    log_covs = torch.log(p_covs)
    log_det_cov = (torch.sum(log_covs, 3)).unsqueeze(3)
    exponent = (-1/2)*((k**2).sum(dim=3, keepdim=True) - 2*(q_mus.sum(dim=3, keepdim=True) - 6*p_sigmas) + (1/d)*(q_mus.sum(dim=3, keepdim=True) - 6*p_sigmas))#(-1/2)*((k-q_mus)*(1/covs)*(k-q_mus)).sum(dim=3, keepdim=True)

    p_ws = p_ws.double()
    log_det_cov = log_det_cov.double()
    exponent = exponent.double()

    #multiplier = ((2*np.pi)**(-latent_variables/2))*np.e**((-1/2)*log_det_cov)
    #ignore first part of multiplier because it cancels out in mc approximation anyways
    
    multiplier = np.e**((-1/2)*log_det_cov)
    multiplier = multiplier.double()

    pdf = multiplier*np.e**exponent
    #since ours is a mixture of gaussians, we sum the pdf over the number of gaussians we have 
    #multiplied by the weight each gaussian has
    pdf = ((p_ws*pdf).sum(dim=1,keepdim=True))
    return pdf

def gimp_loss_pp_mse(outputs):
    # Reconstruction error, log[p(x|z)]
    # Sum over features
    re = F.mse_loss(outputs['x'].unsqueeze(1),outputs['x_hat_all'])*1000
    re = re.view(re.size(0), -1).sum(1)

    # Regularization error: 
    # Kulback-Leibler divergence between approximate posterior, q(z|x)
    # and prior p(z) = N(z | mu, sigma*I).
    n_points = outputs['mc_points']
    
    #load prior and posterior
    p_w, p_mus, p_sigmas = outputs['p']
    q_w, q_mus, q_sigmas = outputs['q']
    
    batch_size = q_w.shape[0]
    d = q_mus.shape[-1]
    
    k_i = sample_from_gmm(n_points, q_w, q_mus, q_sigmas, outputs['cuda'])
    
    q_k_i = get_pdf_all(k_i, q_w, q_mus, q_sigmas, outputs['cuda'])
    p_k_i = get_pdf_all(k_i, p_w, p_mus, p_sigmas, outputs['cuda'])
    #p_k_i = get_pdf_all_weird_prior(k_i, p_w, q_mus, p_sigmas, outputs['cuda'])
    
    with torch.no_grad():
        ep2 = 1e-300
    q_k_i += ep2 
    p_k_i += ep2 

    diff = torch.log(q_k_i) - torch.log(p_k_i)

    kl_approx = diff.mean()
    kl_approx = torch.abs(kl_approx)
    
    if outputs['cuda']:
        kl_approx = kl_approx.type(torch.cuda.FloatTensor)
    else:
        kl_approx = kl_approx.type(torch.FloatTensor)

    MELO =  - kl_approx -torch.mean(re)

    #https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    
    n_q_gaussians = q_w.shape[1]
    n_p_gaussians = p_w.shape[1]
    if (n_q_gaussians == 1) & (n_p_gaussians == 1):
        print('kl approx: ',kl_approx.item())
        cov_q = q_sigmas**2
        cov_p = p_sigmas**2

        a = torch.sum((1/cov_p)*cov_q, dim=2, keepdim=True)
        b = torch.sum((p_mus - q_mus)*(1/cov_p)*(p_mus - q_mus), dim=2, keepdim=True)
        c = -d
        det_sp = (torch.prod(cov_p, 2)).unsqueeze(2)
        det_sq = (torch.prod(cov_q, 2)).unsqueeze(2)
        e = torch.log(det_sp) - torch.log(det_sq)
        kl_div_true = 0.5*(a + b + c + e).mean()
        print('actual true kl                          : ',kl_div_true.item())
    # notice minus sign as we want to maximise ELBO

    return -MELO, [kl_approx, torch.mean(re)]
    
def gimp_loss_pp(outputs):
    # Reconstruction error, log[p(x|z)]
    # Sum over features
    re = binary_cross_entropy(outputs['x_hat_all'], outputs['x'].unsqueeze(1).repeat(1,outputs['num_samples'],1,1,1), reduction="none")#F.mse_loss(outputs['x'].unsqueeze(1),outputs['x_hat_all'])*1000
    re = re.view(re.size(0), -1).sum(1)

    # Regularization error: 
    # Kulback-Leibler divergence between approximate posterior, q(z|x)
    # and prior p(z) = N(z | mu, sigma*I).
    n_points = outputs['mc_points']
    
    #load prior and posterior
    p_w, p_mus, p_sigmas = outputs['p']
    q_w, q_mus, q_sigmas = outputs['q']
    
    batch_size = q_w.shape[0]
    d = q_mus.shape[-1]
    
    k_i = sample_from_gmm(n_points, q_w, q_mus, q_sigmas, outputs['cuda'])
    
    q_k_i = get_pdf_all(k_i, q_w, q_mus, q_sigmas, outputs['cuda'])
    p_k_i = get_pdf_all(k_i, p_w, p_mus, p_sigmas, outputs['cuda'])
    #p_k_i = get_pdf_all_weird_prior(k_i, p_w, q_mus, p_sigmas, outputs['cuda'])
    
    with torch.no_grad():
        ep2 = 1e-300
    q_k_i += ep2 
    p_k_i += ep2 

    diff = torch.log(q_k_i) - torch.log(p_k_i)

    kl_approx = diff.mean()
    kl_approx = torch.abs(kl_approx)
    
    if outputs['cuda']:
        kl_approx = kl_approx.type(torch.cuda.FloatTensor)
    else:
        kl_approx = kl_approx.type(torch.FloatTensor)

    MELO =  - kl_approx -torch.mean(re)

    #https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    
    n_q_gaussians = q_w.shape[1]
    n_p_gaussians = p_w.shape[1]
    if (n_q_gaussians == 1) & (n_p_gaussians == 1):
        print('kl approx: ',kl_approx.item())
        cov_q = q_sigmas**2
        cov_p = p_sigmas**2

        a = torch.sum((1/cov_p)*cov_q, dim=2, keepdim=True)
        b = torch.sum((p_mus - q_mus)*(1/cov_p)*(p_mus - q_mus), dim=2, keepdim=True)
        c = -d
        det_sp = (torch.prod(cov_p, 2)).unsqueeze(2)
        det_sq = (torch.prod(cov_q, 2)).unsqueeze(2)
        e = torch.log(det_sp) - torch.log(det_sq)
        kl_div_true = 0.5*(a + b + c + e).mean()
        print('actual true kl                          : ',kl_div_true.item())
    # notice minus sign as we want to maximise ELBO

    return -MELO, [kl_approx, torch.mean(re)]

def gimp_loss_ub(outputs):
    #use upper bound approximation
    # Reconstruction error, log[p(x|z)]
    # Sum over features
    re = torch.zeros([1,1])#F.mse_loss(outputs['x'].unsqueeze(1),outputs['x_hat_all'])#*10000

    # Regularization error: 
    # Kulback-Leibler divergence between approximate posterior, q(z|x)
    # and prior p(z) = N(z | mu, sigma*I).
    
    #load prior and posterior
    p_w, p_mus, p_sigmas = outputs['p']
    q_w, q_mus, q_sigmas = outputs['q']
    
    batch_size = q_w.shape[0]
    d = q_mus.shape[-1]
    
    kl_approx = upper_bound_approx(outputs['p'],outputs['q'])
    
    if outputs['cuda']:
        kl_approx = kl_approx.type(torch.cuda.FloatTensor)
    else:
        kl_approx = kl_approx.type(torch.FloatTensor)

    MELO =  - kl_approx #-torch.mean(re)

    #https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    
    n_q_gaussians = q_w.shape[1]
    n_p_gaussians = p_w.shape[1]
    if (n_q_gaussians == 1) & (n_p_gaussians == 1):
        print('kl approx: ',kl_approx.item())
        cov_q = q_sigmas**2
        cov_p = p_sigmas**2

        a = torch.sum((1/cov_p)*cov_q, dim=2, keepdim=True)
        b = torch.sum((p_mus - q_mus)*(1/cov_p)*(p_mus - q_mus), dim=2, keepdim=True)
        c = -d
        det_sp = (torch.prod(cov_p, 2)).unsqueeze(2)
        det_sq = (torch.prod(cov_q, 2)).unsqueeze(2)
        e = torch.log(det_sp) - torch.log(det_sq)
        kl_div_true = 0.5*(a + b + c + e).mean()
        print('actual true kl                          : ',kl_div_true.item())
    # notice minus sign as we want to maximise ELBO

    return -MELO, [kl_approx, torch.mean(re)]


def gimp_loss_mse(outputs):
    # Reconstruction error, log[p(x|z)]
    # Sum over features
    re = F.mse_loss(outputs['x'],outputs['x_hat'])*1000

    # Regularization error: 
    # Kulback-Leibler divergence between approximate posterior, q(z|x)
    # and prior p(z) = N(z | mu, sigma*I).
    
    n_points = outputs['mc_points']
    
    #load prior
    g_w, g_mus, g_sigmas = outputs['g']
    
    #load posterior
    f_w = outputs['w']
    f_mus = outputs['mus'].unsqueeze(3).repeat(1,1,1,n_points)
    f_sigmas = outputs['sigmas'].unsqueeze(3).repeat(1,1,1,n_points)
    
    
    batch_size = f_w.shape[0]
    n_q_gaussians = f_w.shape[1]
    n_p_gaussians = g_w.shape[1]
    d = f_mus.shape[-2]
    
    with torch.no_grad():
        chosen_gaussians = torch.multinomial(f_w,n_points, replacement=True).view(batch_size, 1, 1, n_points).repeat(1,1,d,1)
        ep = torch.randn(batch_size, 1, d, n_points)
        
        if outputs['cuda']:
            chosen_gaussians = chosen_gaussians.cuda()
            ep = ep.cuda()


    
    #sample n_points from the f function, these samples are k_i
    k_mu = torch.gather(f_mus, 1, chosen_gaussians)
    k_sigma = torch.gather(f_sigmas, 1, chosen_gaussians)
    k_i = k_mu + ep*k_sigma
    
    
    f_k_i = get_pdf(k_i.repeat(1,n_q_gaussians,1,1), f_w, f_mus, f_sigmas, outputs['cuda'])
    g_k_i = get_pdf(k_i.repeat(1,n_p_gaussians,1,1), g_w.repeat(batch_size,1,1,1), g_mus.repeat(batch_size,1,1,1), g_sigmas.repeat(batch_size,1,1,1), outputs['cuda'])
    
    with torch.no_grad():
        ep2 = 1e-300
        
    f_k_i += ep2 
    g_k_i += ep2 
    

    diff = torch.log(f_k_i) - torch.log(g_k_i)

    #compute the kl divergence between f||g
    kl_approx = diff.mean()
    
    kl_approx = torch.abs(kl_approx)
    
    if outputs['cuda']:
        kl_approx = kl_approx.type(torch.cuda.FloatTensor)
    else:
        kl_approx = kl_approx.type(torch.FloatTensor)

    # Combining the two terms in the evidence lower bound objective (ELBO) 
    # mean over batch
    #print('losses : ',-torch.mean(likelihood),  kl_approx)
    MELO = -torch.mean(re) - kl_approx
    #print(kl_approx)
    #https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    #since kl(f||g), f is 0, g is 1
    
    if n_gaussians == 1:
        print('kl approx: ',kl_approx.item())
        cov_f = f_sigmas**2
        cov_g = g_sigmas**2

        a = torch.sum((1/cov_g)*cov_f, dim=2, keepdim=True)
        b = torch.sum((g_mus - f_mus)*(1/cov_g)*(g_mus - f_mus), dim=2, keepdim=True)
        c = -d
        det_sg = (torch.prod(cov_g, 2)).unsqueeze(2)
        det_sf = (torch.prod(cov_f, 2)).unsqueeze(2)
        e = torch.log(det_sg) - torch.log(det_sf)
        kl_div_true = 0.5*(a + b + c + e).mean()
        print('actual true kl                          : ',kl_div_true.item())
    # notice minus sign as we want to maximise ELBO

    return -MELO, [kl_approx, torch.mean(re)]

def gimp_loss(outputs):
    
    # Reconstruction error, log[p(x|z)]
    # Sum over features
    likelihood = -binary_cross_entropy(outputs['x_hat'], outputs['x'], reduction="none")
    likelihood = likelihood.view(likelihood.size(0), -1).sum(1)

    # Regularization error: 
    # Kulback-Leibler divergence between approximate posterior, q(z|x)
    # and prior p(z) = N(z | mu, sigma*I).
    
    n_points = outputs['mc_points']
    
    #load prior
    g_w, g_mus, g_sigmas = outputs['g']
    
    #load posterior
    f_w = outputs['w']
    f_mus = outputs['mus'].unsqueeze(3).repeat(1,1,1,n_points)
    f_sigmas = outputs['sigmas'].unsqueeze(3).repeat(1,1,1,n_points)
    
    
    batch_size = f_w.shape[0]
    n_gaussians = f_w.shape[1]
    d = f_mus.shape[-2]
    
    with torch.no_grad():
        chosen_gaussians = torch.multinomial(f_w,n_points, replacement=True).view(batch_size, 1, 1, n_points).repeat(1,1,d,1)
        ep = torch.randn(batch_size, 1, d, n_points)
        
        if outputs['cuda']:
            chosen_gaussians = chosen_gaussians.cuda()
            ep = ep.cuda()


    
    #sample n_points from the f function, these samples are k_i
    k_mu = torch.gather(f_mus, 1, chosen_gaussians)
    k_sigma = torch.gather(f_sigmas, 1, chosen_gaussians)
    k_i = k_mu + ep*k_sigma
    
    f_k_i = get_pdf(k_i, f_w, f_mus, f_sigmas, outputs['cuda'])
    g_k_i = get_pdf(k_i, g_w, g_mus, g_sigmas, outputs['cuda'])
    
    with torch.no_grad():
        ep2 = 1e-300
        
    f_k_i += ep2 
    g_k_i += ep2 
    

    diff = torch.log(f_k_i) - torch.log(g_k_i)

    #compute the kl divergence between f||g
    kl_approx = diff.mean()
    
    kl_approx = torch.abs(kl_approx)
    
    if outputs['cuda']:
        kl_approx = kl_approx.type(torch.cuda.FloatTensor)
    else:
        kl_approx = kl_approx.type(torch.FloatTensor)

    # Combining the two terms in the evidence lower bound objective (ELBO) 
    # mean over batch
    #print('losses : ',-torch.mean(likelihood),  kl_approx)
    MELO = torch.mean(likelihood) - kl_approx
    #print(kl_approx)
    #https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    #since kl(f||g), f is 0, g is 1
    
    if n_gaussians == 1:
        print('kl approx: ',kl_approx.item())
        cov_f = f_sigmas**2
        cov_g = g_sigmas**2

        a = torch.sum((1/cov_g)*cov_f, dim=2, keepdim=True)
        b = torch.sum((g_mus - f_mus)*(1/cov_g)*(g_mus - f_mus), dim=2, keepdim=True)
        c = -d
        det_sg = (torch.prod(cov_g, 2)).unsqueeze(2)
        det_sf = (torch.prod(cov_f, 2)).unsqueeze(2)
        e = torch.log(det_sg) - torch.log(det_sf)
        kl_div_true = 0.5*(a + b + c + e).mean()
        print('actual true kl                          : ',kl_div_true.item())
    # notice minus sign as we want to maximise ELBO

    return -MELO, [kl_approx, -torch.mean(likelihood)]


def chello_loss(outputs):
    
    n_points = 10000
    
    #load prior
    g_w, g_mus, g_sigmas = outputs['g']
    
    #load posterior
    f_w = outputs['w'] # [batch_size, n_gaussians]
    f_mus = outputs['mus'].unsqueeze(3).repeat(1,1,1,n_points)
    f_sigmas = outputs['sigmas'].unsqueeze(3).repeat(1,1,1,n_points)
    
    
    batch_size = f_w.shape[0]
    n_gaussians = f_w.shape[1]
    d = f_mus.shape[-2]
    num_samples = outputs['x_hat_all'].shape[1]
    
    
    # Reconstruction error, log[p(x|z)]
    # Sum over features
    likelihood = -binary_cross_entropy(outputs['x_hat'], outputs['x'], reduction="none")
    likelihood = likelihood.view(likelihood.size(0), -1).sum(1)

    #atkins - weight loss
    reconstruction_loss = torch.mean((outputs['x_hat_all'] - outputs['x'].unsqueeze(1)) ** 2, dim=(2,3,4))

    
    chosen_w = outputs['chosen_w'] #[batch_size, num_samples, 1, 1]
    
    with torch.no_grad():
        atkins = torch.zeros((batch_size, n_gaussians))
        if outputs['cuda']:
            atkins = atkins.cuda()

        for idx in range(batch_size):
            lower_counts = np.zeros(n_gaussians)
            higher_counts = np.zeros(n_gaussians)
            chosen_w_1 = chosen_w.squeeze(dim=3).squeeze(dim=2).unsqueeze(1).repeat(1,num_samples,1)[idx]
            chosen_w_2 = chosen_w.squeeze(dim=3).squeeze(dim=2).unsqueeze(2).repeat(1,1,num_samples)[idx]
            atkins_1 = reconstruction_loss.unsqueeze(1).repeat(1,num_samples,1)[idx]
            atkins_2 = reconstruction_loss.unsqueeze(2).repeat(1,1,num_samples)[idx]

            mask = (chosen_w_1 != chosen_w_2).triu()

            lower = torch.where((atkins_1 < atkins_2)[mask], chosen_w_1[mask], chosen_w_2[mask])
            higher = torch.where((atkins_1 >= atkins_2)[mask], chosen_w_1[mask], chosen_w_2[mask])
            unique, counts = np.unique(lower.cpu().detach(), return_counts=True)
            lower_counts[unique] = counts
            unique, counts = np.unique(higher.cpu().detach(), return_counts=True)
            higher_counts[unique] = counts
            res = lower_counts/(lower_counts + higher_counts)
            res = np.where(np.isnan(res), outputs["w"][idx].cpu().detach().numpy(), res)
            
            print('What we want weights to be: ', torch.softmax(torch.tensor(res), dim=0))
            print('What they actually are: ', outputs["w"][idx])
            atkins[idx] = torch.tensor(res)
    
        atkins = torch.softmax(atkins,dim=1)
        
    atkins_loss = ((atkins - outputs['w'])**2).mean()
    
    
    # Regularization error: 
    # Kulback-Leibler divergence between approximate posterior, q(z|x)
    # and prior p(z) = N(z | mu, sigma*I).
    
    with torch.no_grad():
        chosen_gaussians = torch.multinomial(f_w,n_points, replacement=True).view(batch_size, 1, 1, n_points).repeat(1,1,d,1)
        ep = torch.randn(batch_size, 1, d, n_points)
        
        if outputs['cuda']:
            chosen_gaussians = chosen_gaussians.cuda()
            ep = ep.cuda()
        

    
    #sample n_points from the f function, these samples are x_i
    x_mu = torch.gather(f_mus, 1, chosen_gaussians)
    x_sigma = torch.gather(f_sigmas, 1, chosen_gaussians)
    x_i = x_mu + ep*x_sigma    
    
    #crucially notice that g_w is used here, not the actual f_w
    f_x_i = get_pdf(x_i, g_w, f_mus, f_sigmas, outputs['cuda'])
    g_x_i = get_pdf(x_i, g_w, g_mus, g_sigmas, outputs['cuda'])
    
    with torch.no_grad():
        ep2 = 1e-40
        
    f_x_i += ep2 
    g_x_i += ep2 
    

    diff = torch.log(f_x_i) - torch.log(g_x_i)

    kl_approx = diff.mean()
    
    kl_approx = torch.abs(kl_approx)
    
    if outputs['cuda']:
        kl_approx = kl_approx.type(torch.cuda.FloatTensor)
    else:
        kl_approx = kl_approx.type(torch.FloatTensor)

    # Combining the two terms in the evidence lower bound objective (ELBO) 
    # mean over batch
    CHELLO = torch.mean(likelihood) - atkins_loss*10 - kl_approx
    #print(kl_approx)
    if n_gaussians == 1:
        print('kl approx: ',kl_approx.item())
        cov_f = f_sigmas**2
        cov_g = g_sigmas**2

        a = torch.sum((1/cov_g)*cov_f, dim=2, keepdim=True)
        b = torch.sum((g_mus - f_mus)*(1/cov_g)*(g_mus - f_mus), dim=2, keepdim=True)
        c = -d
        det_sg = (torch.prod(cov_g, 2)).unsqueeze(2)
        det_sf = (torch.prod(cov_f, 2)).unsqueeze(2)
        e = torch.log(det_sg/det_sf)
        kl_div_true = 0.5*(a + b + c + e).mean()
        print('actual true kl                          : ',kl_div_true.item())
    # notice minus sign as we want to maximise ELBO
    return -CHELLO, [ atkins_loss*10, kl_approx, -torch.mean(likelihood)]

def dumbo_loss(outputs):
    #Try to update mean based on epsilon which led to the best reconstruction
    
    #epsilon mean update loss
    
    
    # Reconstruction error, log[p(x|z)]
    # Sum over features
    likelihood = -binary_cross_entropy(outputs['x_hat'], outputs['x'], reduction="none")
    likelihood = likelihood.view(likelihood.size(0), -1).sum(1)

    
    # Regularization error: 
    # Kulback-Leibler divergence between approximate posterior, q(z|x)
    # and prior p(z) = N(z | mu, sigma*I).
    
    # In the case of the KL-divergence between diagonal covariance Gaussian and 
    # a standard Gaussian, an analytic solution exists. Using this excerts a lower
    # variance estimator of KL(q||p)
    kl = -0.5 * torch.sum(1 + outputs['log_var'] - outputs['mu']**2 - torch.exp(outputs['log_var']), dim=1)

    # Combining the two terms in the evidence lower bound objective (ELBO) 
    # mean over batch
    ELBO = torch.mean(likelihood) - torch.mean(kl)
    
    x = outputs['x'].unsqueeze(1)
    x_hat_all = outputs['x_hat_all']
    ret = (x_hat_all - x) ** 2
    avg_epsilon = outputs['epsilon'][[i for i in range(x.size()[0])],torch.argmin(torch.mean(ret,dim=(2,3,4)), dim=1)]
    updated_mu = outputs['mu'].clone() + avg_epsilon
    mu_mse = F.mse_loss(updated_mu, outputs['mu'])
    
    DUMBO = mu_mse - ELBO
    # notice minus sign as we want to maximise ELBO
    return DUMBO, [mu_mse, torch.mean(kl), -torch.mean(likelihood)]

def elbo_loss(outputs):
    # Reconstruction error, log[p(x|z)]
    # Sum over features
    likelihood = -binary_cross_entropy(outputs['x_hat'], outputs['x'], reduction="none")
    likelihood = likelihood.view(likelihood.size(0), -1).sum(1)

    # Regularization error: 
    # Kulback-Leibler divergence between approximate posterior, q(z|x)
    # and prior p(z) = N(z | mu, sigma*I).
    
    # In the case of the KL-divergence between diagonal covariance Gaussian and 
    # a standard Gaussian, an analytic solution exists. Using this excerts a lower
    # variance estimator of KL(q||p)
    kl = -0.5 * torch.sum(1 + outputs['log_var'] - outputs['mu']**2 - torch.exp(outputs['log_var']), dim=1)

    # Combining the two terms in the evidence lower bound objective (ELBO) 
    # mean over batch
    ELBO = torch.mean(likelihood) - torch.mean(kl)
    
    # notice minus sign as we want to maximise ELBO
    return -ELBO, [torch.mean(kl), -torch.mean(likelihood)]

def elbo_loss_laplace(outputs):
    # Reconstruction error, log[p(x|z)]
    # Sum over features
    likelihood = -binary_cross_entropy(outputs['x_hat'], outputs['x'], reduction="none")
    likelihood = likelihood.view(likelihood.size(0), -1).sum(1)

    # Regularization error: 
    # Kulback-Leibler divergence between approximate posterior, q(z|x)
    # and prior p(z) = N(z | mu, sigma*I).
    
    # In the case of the KL-divergence between diagonal covariance Gaussian and 
    # a standard Gaussian, an analytic solution exists. Using this excerts a lower
    # variance estimator of KL(q||p)
    b = outputs['b']
    
    #Taken from https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html
    scale_ratio = b/1
    loc_abs_diff = torch.abs(outputs['mu'] - 0)
    t1 = -(scale_ratio.log())
    t2 = loc_abs_diff/1
    t3 = scale_ratio*torch.exp(-loc_abs_diff/b)
    kl = t1 + t2 + t3 - 1
    
    # Combining the two terms in the evidence lower bound objective (ELBO) 
    # mean over batch
    ELBO = torch.mean(likelihood) - torch.mean(kl)
    
    # notice minus sign as we want to maximise ELBO
    return -ELBO, [torch.mean(kl), -torch.mean(likelihood)]

def rand(w, mus, covs):
    chosen_gaussian = np.random.choice(range(len(w)),p=w)
    return multivariate_normal.rvs(mus[chosen_gaussian], np.diag(covs[chosen_gaussian]))

def pdf(x, w, mus, covs):
    p = 0
    for i in range(len(w)):
        p += w[i]*multivariate_normal.pdf(x, mean=mus[i], cov=np.diag(covs[i]))
    return p

#def gaussian_noise(x, coeff=0.2):
#    if x.is_cuda:
#        return torchvision.transforms.Lambda(lambda x: x + torch.cuda.FloatTensor(x.size()).normal_()*coeff)
#    else:
#        return torchvision.transforms.Lambda(lambda x: x + torch.FloatTensor(x.size()).normal_()*coeff)

def create_mask(x, p=0.2):
    batchSize, channels, h, w = x.size()
    maskSize = torch.Size((batchSize, 1, h, w))
    if x.is_cuda:
        return (torch.cuda.FloatTensor(maskSize).uniform_() >= 0.2).type(torch.cuda.FloatTensor)
    else:
        return (torch.FloatTensor(maskSize).uniform_() >= 0.2).type(torch.FloatTensor)

#def masking_noise(p=0.2):
#    assert p >= 0
#    assert p < 1
#    return torchvision.transforms.Lambda(lambda x: x*create_mask(x, p=p))

class masking_noise(object):
    def __init__(self, coeff=0.2):
        assert coeff >= 0
        assert coeff < 1
        self.coeff = coeff
        
    def __call__(self, x):
        return x*create_mask(x, p=self.coeff)
        
class gaussian_noise(object):
    def __init__(self, coeff=0.2):
        self.coeff = coeff
        
    def __call__(self, x):
        if x.is_cuda:
            return x + torch.cuda.FloatTensor(x.size()).normal_()*self.coeff
        else:
            return x + torch.FloatTensor(x.size()).normal_()*self.coeff

loss_function_dict = {
    'mse': mse_loss,
    'bce': bce_loss,
    'elbo': elbo_loss,
    'elbo_laplace': elbo_loss_laplace,
    'dumbo': dumbo_loss,
    'melo': gimp_loss,
    'gimp': gimp_loss,
    'gimp_mse': gimp_loss_mse,
    'chello': chello_loss,
    'gimp_pp': gimp_loss_pp,
    'gimp_pp_mse': gimp_loss_pp_mse,
}
optimizer_dict = {
    'adam': optim.Adam,
    'sgd': optim.SGD,
    'rmsprop': optim.RMSprop
}

transforms_dict = {
    'resize': myTransforms.Resize,
    'hflip': myTransforms.RandomHorizontalFlip,
    'totensor': myTransforms.ToTensor,
    'normalize': torchvision.transforms.Normalize, #mean, std, inplace=False
    'colorjitter': myTransforms.ColorJitter, # brightness=0, contrast=0, saturation=0, hue=0
    #'affine': myTransforms.RandomAffine, #degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0
    'rotateandscale': myTransforms.RandomRotationandScaling, #degrees, resample=False, expand=False, center=None
    'gaussian_noise': gaussian_noise,
    'masking_noise': masking_noise,
    'topilimage': torchvision.transforms.ToPILImage,
    'grayscale': myTransforms.Grayscale,
}

datagetter_dict = {
    'butterfly': datagetters.butterfly,
    'toy': datagetters.toy,
    'mnist': datagetters.mnist
}

