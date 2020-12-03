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
from loss_functions import loss_l2, weighted_cross_entropy
from SimpleDLA import *

# -----------------------------------------------------------

def get_lr(epoch, lr0, gamma):
    return lr0*gamma**epoch 
# -----------------------------------------------------------
    
def get_momentum(epoch, p_i, p_f, T):
    if epoch<T:
        p = (epoch/T)*p_f + (1 - (epoch/T))*p_i
    else:
        p = p_f
    
    return p


#Transforms for SimpleDLA
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)



'''
#transforms from first source for CNN

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True)

testset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False)

s=torch.utils.data.RandomSampler(trainset,num_samples = int(len(trainset)*0.1),replacement=True)
sub = torch.utils.data.Subset(trainset,s)
train_loader = torch.utils.data.DataLoader(s, batch_size=32, shuffle=True)
'''

data_pct = 0.1
ntrainval = len(trainset)
TRAIN_PCT = 0.8
ndata = int(ntrainval*data_pct)
ntrain = int(ndata*TRAIN_PCT)
slimmed_data,rest_data =torch.utils.data.random_split(trainset,[ndata,ntrainval-ndata],generator=torch.Generator().manual_seed(42))
slim_train_data,slim_val_data = torch.utils.data.random_split(slimmed_data,[ntrain,ndata-ntrain],generator=torch.Generator().manual_seed(42))
slim_train_loader = torch.utils.data.DataLoader(slim_train_data, batch_size=32, shuffle=True)

slim_test_data,rest_test_data =torch.utils.data.random_split(testset,[int(data_pct*len(testset)),len(testset)-int(data_pct*len(testset))],generator=torch.Generator().manual_seed(42))

# counts = [0 for i in range(10)]
# for i in range(10):
#     counts[i] = sum([sum((y==i).tolist()) for b,(x,y) in enumerate(slim_train_loader)])

# print(counts)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


params = {'epochs': [200],
          'batch_size' : [32],
          'lr' : [1e-1],
          'decay' :[1e-2],
          'gamma' : [0.8],
          'step_size' :[5],
          'num_bins' : [50],
          'prune_milestones' : [[1000]],
          'prune_sigmas' : [[1]],
          'prune_gamma' : [10],
          'ignore_below' : [True],
          'p_i' : [5e-1],
          'p_f' : [0.99],
          'T' : [200]
          }


keys, values = zip(*params.items())
prune_param_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]


# model= LeNet_KG(in_chan=3, out_chan=10, imsize=32, kernel_size=5)
# results = param_search(model,params,slim_train_data,slim_val_data,slim_test_data)

# min_params = ast.literal_eval(min(results.keys(),key = lambda k: results[k]['num_errors']))

param_dict = {'epochs': 200,
          'batch_size' : 32,
          'lr' : 1e-2,
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


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

epochs        = 200   # number of training epochs
valid_size    = 50    # number of samples for validation
batch_size    = 50    # number of samples per mini-batch
imsize        = 50    # image size
num_classes   = 2     # The number of output classes. FRI/FRII
learning_rate = 1e-3  # The speed of convergence
momentum      = 9e-1  # momentum for optimizer
decay         = 1e-6  # weight decay for regularisation
random_seed   = 42
p_i = 5e-1
p_f = 0.99
T = 200

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# model= LeNet_KG(in_chan=3, out_chan=10, imsize=32, kernel_size=5)
model = SimpleDLA()
# model, df =  train(model,slim_train_data,slim_val_data,param_dict)

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=p_i, dampening=p_i,weight_decay=5e-4)

# -----------------------------------------------------------------------------

# summary(model, (1, imsize, imsize))

# -----------------------------------------------------------------------------



use_cuda = False

if use_cuda:
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True



epoch_trainaccs, epoch_testaccs = [], []
epoch_trainloss, epoch_testloss = [], []
for epoch in range(epochs):  # loop over the dataset multiple times

    model.train()
    train_losses, train_accs = [], []; acc = 0
    for batch, (x_train, y_train) in enumerate(trainloader):
        print('Batch: ',batch,end='\r')
        
        if use_cuda:
            x_train = x_train.cuda()
            y_train = y_train.cuda()
        
        model.zero_grad()

        pred, std = model(x_train)
        # add line in here to call sample_sigma()
        # d, yd = model.sample_sigma(pred,std,y_train)
        loss = weighted_cross_entropy(pred,std,y_train)
        loss.backward()
        optimizer.step()

        acc = (pred.argmax(dim=-1) == y_train).to(torch.float32).mean()
        train_accs.append(acc.mean().item())
        train_losses.append(loss.item())


    with torch.no_grad():
        model.eval()
        test_losses, test_accs = [], []; acc = 0
        for i, (x_test, y_test) in enumerate(testloader):
            test_pred, test_std = model(x_test)
            # test_d, test_yd = model.sample_sigma(pred,std,y_train)
        
            test_loss = weighted_cross_entropy(test_pred,test_std,y_test)
            acc = (test_pred.argmax(dim=-1) == y_test).to(torch.float32).mean()
            test_losses.append(loss.item())
            test_accs.append(acc.mean().item())

    print('Epoch: {}, Loss: {}, Accuracy: {}'.format(epoch, np.mean(test_losses), np.mean(test_accs)))
    epoch_trainaccs.append(np.mean(train_accs))
    epoch_testaccs.append(np.mean(test_accs))
    epoch_trainloss.append(np.mean(train_losses))
    epoch_testloss.append(np.mean(test_losses))
    
    
    optimizer.param_groups[0]['lr'] = get_lr(epoch, learning_rate,gamma)
    optimizer.param_groups[0]['momentum'] = get_momentum(epoch,p_i,p_f, T)
    optimizer.param_groups[0]['dampening'] = get_momentum(epoch, p_i, p_f,T)

print("Final test error: ",100.*(1 - epoch_testaccs[-1]))

# save trained model:
outfile = "./mb_lenet_kg.pt"
torch.save(model.state_dict(), outfile)









