import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
#from torchsummary import summary

import numpy as np
import PIL

from models import LeNet, LeNet_KG, CNN

from training import train, train_prune, standard_train
import matplotlib.pyplot as plt
import csv
import os
import ast

import pandas as pd

from parameter_search import param_search
import itertools
import random

from pruner import Pruner


transform = transforms.Compose([transforms.ToTensor()])


trainset = datasets.MNIST('./MNIST', download=True, train=True, transform=transform)
valset = datasets.MNIST('./MNIST', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=True)


# data_pct = 0.1
# ntrainval = len(trainset)
# TRAIN_PCT = 0.8
# ndata = int(ntrainval*data_pct)
# ntrain = int(ndata*TRAIN_PCT)
# slimmed_data,rest_data =torch.utils.data.random_split(trainset,[ndata,ntrainval-ndata],generator=torch.Generator().manual_seed(42))
# slim_train_data,slim_val_data = torch.utils.data.random_split(slimmed_data,[ntrain,ndata-ntrain],generator=torch.Generator().manual_seed(42))
# slim_train_loader = torch.utils.data.DataLoader(slim_train_data, batch_size=32, shuffle=True)

# slim_test_data,rest_test_data =torch.utils.data.random_split(testset,[int(data_pct*len(testset)),len(testset)-int(data_pct*len(testset))],generator=torch.Generator().manual_seed(42))

# counts = [0 for i in range(10)]
# for i in range(10):
#     counts[i] = sum([sum((y==i).tolist()) for b,(x,y) in enumerate(slim_train_loader)])

# print(counts)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# params = {'epochs': [200],
#           'batch_size' : [32],
#           'lr' : [1e-1],
#           'decay' :[1e-2],
#           'gamma' : [0.8],
#           'step_size' :[5],
#           'num_bins' : [50],
#           'prune_milestones' : [[1000]],
#           'prune_sigmas' : [[1]],
#           'prune_gamma' : [10],
#           'ignore_below' : [True],
#           'p_i' : [5e-1],
#           'p_f' : [0.99],
#           'T' : [200]
#           }


# keys, values = zip(*params.items())
# prune_param_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]


# model= LeNet_KG(in_chan=3, out_chan=10, imsize=32, kernel_size=5)
# results = param_search(model,params,slim_train_data,slim_val_data,slim_test_data)

# min_params = ast.literal_eval(min(results.keys(),key = lambda k: results[k]['num_errors']))

param_dict = {'epochs': 50,
          'batch_size' : 128,
          'lr' : 1e-3,
          'decay' : 1e-2,
          'gamma' : 0.8,
          'step_size' : 5,
          'num_bins' : 50,
          'prune_milestones' : [1000],
          'prune_sigmas' : [1],
          'prune_gamma' : 10,
          'ignore_below' : True,
          'p_i' : 5e-1,
          'p_f' : 0.99,
          'T' : 100
          }

model= LeNet_KG(in_chan=1, out_chan=10, imsize=28, kernel_size=5)
# model = CNN()
model, df =  train(model,trainset,valset,param_dict)
model.eval()
with torch.no_grad():
    alltrainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    for b,(x,y) in enumerate(alltrainloader):
        v,s = model(x)
    c = v.argmax(dim=1)
    std = torch.empty((s.size()[0]),requires_grad=False)
    for i in range(s.size()[0]):
        std[i] = torch.sqrt(F.softplus(s[i,c[i]]))
