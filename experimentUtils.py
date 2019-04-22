import os
import torch
import torch.optim as optim
import torchvision.transforms
import torch.nn.functional as F

from glob import glob
from torch.nn.functional import binary_cross_entropy

import myTransforms

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

def mse_loss(outputs):
    return F.mse_loss(outputs['x'],outputs['x_hat'])

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
    return -ELBO#, kl.sum()

def gaussian_noise(coeff=0.2):
    return torchvision.transforms.Lambda(lambda x: x + torch.cuda.FloatTensor(x.size()).normal_()*coeff)

def create_mask(x, p=0.2):
    batchSize, channels, h, w = x.size()
    maskSize = torch.Size((batchSize, 1, h, w))
    return (torch.cuda.FloatTensor(maskSize).uniform_() >= 0.2).type(torch.cuda.FloatTensor)

def masking_noise(p=0.2):
    assert p >= 0
    assert p < 1
    return torchvision.transforms.Lambda(lambda x: x*create_mask(x, p=p))

loss_function_dict = {
    'mse': mse_loss,
    'elbo': elbo_loss
}
optimizer_dict = {
    'adam': optim.Adam,
    'sgd': optim.SGD,
    'rmsprop': optim.RMSprop
}

transforms_dict = {
    'resize': torchvision.transforms.Resize,
    'hflip': torchvision.transforms.RandomHorizontalFlip,
    'totensor': torchvision.transforms.ToTensor,
    'normalize': torchvision.transforms.Normalize, #mean, std, inplace=False
    'colorjitter': torchvision.transforms.ColorJitter, # brightness=0, contrast=0, saturation=0, hue=0
    'affine': torchvision.transforms.RandomAffine, #degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0
    'rotate': torchvision.transforms.RandomRotation, #degrees, resample=False, expand=False, center=None
    'gaussian_noise': gaussian_noise,
    'masking_noise': masking_noise,
    'topilimage': torchvision.transforms.ToPILImage
}
