import os
import gc
import random

import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import optim

import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm#_notebook as tqdm

from experiment import experiment, load_experiment
from models.BaseModel import BaseModel
from experimentUtils import get_model_called, get_loss_function_called, get_optimizer_called, get_transform_called

from utils import check_memory_usage, start_timer, tick, count_parameters

import matplotlib
import matplotlib.pyplot as plt

#%matplotlib nbagg
#%matplotlib inline

cuda = torch.cuda.is_available()

#%env CUDA_LAUNCH_BLOCKING=1