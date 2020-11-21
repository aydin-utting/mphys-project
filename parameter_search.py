import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
#from torchsummary import summary

import numpy as np
import PIL
import time
import matplotlib.pyplot as plt
import csv

from models import LeNet, LeNet_KG
from loss_functions import loss_l2
from torch.optim import lr_scheduler
from training import train


from pruner import Pruner

import pandas as pd

import itertools


def param_search(model,params, train_data, val_data, test_data):
    
    # start_state = model.state_dict()
    
    #train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=False)
        
    
    keys, values = zip(*params.items())
    param_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    rem_list = []
    for p in param_dicts:
        if len(p['prune_milestones'])!=len(p['prune_sigmas']):
            rem_list.append(p)
            
    for r in rem_list:
        param_dicts.remove(r)

    
    
    results = {}
    for p in param_dicts:
        print(p)
        # model.load_state_dict(start_state)
        model= LeNet_KG(in_chan=1, out_chan=2, imsize=50, kernel_size=5)
        model, df = train(
            model,
            train_data,
            val_data,
            p)
        
        model.eval()
        
        with torch.no_grad():
            num_errors = 0
            for batch, (x,y) in enumerate(test_loader):
                pred, s  = model(x)
                c = pred.argmax(dim=1)
                num_errors += (c!=y).sum().item()
            
            results[str(p)] = {'model': model, 'df': df ,'num_errors' : num_errors}
            
    return results
    
    