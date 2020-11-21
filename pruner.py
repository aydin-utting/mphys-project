
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


class Pruner():
    
    def __init__(self, dataset,optimizer,params):
        
        self.params = params
        
       
        self.milestones = self.params['prune_milestones']
        self.sigmas = self.params['prune_sigmas']
       
        
        if len(self.milestones) != len(self.sigmas):
            raise ValueError("Different numbers of milestones and sigmas given.")
        
        self.dataset = dataset
       
        self.sigmadict = dict([[i,j] for i,j in zip(self.milestones,self.sigmas)])
        
        self.step_number = 0
        
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=False)
        
        
        self.optimizer = optimizer
        self.re_learner = lr_scheduler.StepLR(self.optimizer,step_size=1,gamma=self.params['prune_gamma'])
        
        self.ignore_list = []
        
        
    def get_pct(self,val_arr):
            sorted_arr = np.sort(val_arr)
            index = int(((1.0-self.sigmadict[self.step_number]) * len(sorted_arr)))
           # print("Sorted_arr:\n{}\n index:{}".format(sorted_arr,index))
            return sorted_arr[index]    
        
    def get_std(self,val_arr, plot=False):
        hist, bin_edges = np.histogram(val_arr,bins=num_bins,density=True)
        if(plot):
            plt.hist(val_arr,bins=self.params(['num_bins']))
            plt.show()
        HM = max(hist)/2
        index = np.argmax(hist)
        mu = bin_edges[index];
        limit = 0
        for i in range(index,len(hist)):
            if(hist[i]<HM):
                sigma = (bin_edges[i]-mu)/1.1775
                return mu, sigma
            
    def get_batch_sigma(self,model,train_loader):
        with torch.no_grad():
                model.eval()
                train_ale_unc = []; acc = 0
                batch_sigma = []
                for i, (x_train, y_train) in enumerate(train_loader):
                    train_pred, train_std = model(x_train)
                    
                    for training_data_point in range(train_std.size()[0]):
                        batch_sigma.append(np.sqrt(F.softplus(train_std[training_data_point,train_pred.argmax(dim=1)[training_data_point].item()]).item()) )
                    
        return batch_sigma
            
    
    def prune_dataloaders(self,model, train_loader):
       
        with torch.no_grad():
            model.eval()
            if(self.step_number in self.milestones):
                unc = []
                for  b,(x,y) in enumerate(train_loader):
                    pred,s = model(x)
                    c = pred.argmax(dim=1)
                    for i in range(len(c)):
                        unc.append(torch.sqrt(F.softplus(s[i,c[i]])).item())
                fig = plt.figure()
                ax = fig.add_subplot()
                ax.hist(unc,100,range=[0,1.],alpha=0.6)
                ax.set_title('Aleatoric Uncertainty distribution of training data before and after pruning')
                
                self.ignore_list = []
    
                
                # mu, sigma = self.get_std(self.get_batch_sigma(model,train_loader))
                
                # max_unc = np.array([self.get_pct(self.get_batch_sigma(model,train_loader)) for i in range(50)]).mean()
                # print(max_unc)
               
                self.forward_add_ignore_average(model,50)
               
                # max_unc = self.get_pct(self.get_batch_sigma(model,train_loader))
                # for b, (x,y) in enumerate(self.dataloader):
                    #self.forward_add_ignore(model,x,mu-self.sigmadict[self.step_number]*sigma)
                    # self.forward_add_ignore(model, x,max_unc)
                    
                train_loader, ignore_loader = self.create_dataloaders()
                
                
                unc = []
                for  b,(x,y) in enumerate(train_loader):
                    pred,s = model(x)
                    c = pred.argmax(dim=1)
                    for i in range(len(c)):
                        unc.append(torch.sqrt(F.softplus(s[i,c[i]])).item())
                ax.hist(unc,100,range=[0,1.],alpha=0.6)
                fig.show()
                
                print(sum([len(y) for b,(x,y) in enumerate(train_loader)]))
                self.re_learner.step()
                
                # for param_group in self.optimizer.param_groups:
                #     param_group['lr'] = lr
                
        self.step_number += 1
        return train_loader
        
        
    def forward_add_ignore(self, model, x, max_unc):
        with torch.no_grad():
            model.eval()
            v,s = model.forward(x)
            c = v.argmax(dim=1)
            std = torch.sqrt(F.softplus(s))
            for i in range(x.size()[0]):
                if std[i,c[i]] > max_unc and (i not in self.ignore_list):
                    self.ignore_list.append(i)
                    
    def forward_add_ignore_average(self, model,num_samples,max_unc=None):
        alea_unc = torch.empty((len(self.dataset),num_samples))
        for n in range(num_samples):        
            with torch.no_grad():
                model.eval()
                for b,(x,y) in enumerate(self.dataloader):
                    v,s = model.forward(x)
                    c = v.argmax(dim=1)
                    std = torch.sqrt(F.softplus(s))
                    for i in range(x.size()[0]):
                        alea_unc[i,n] = std[i,c[i]]
    
        max_unc = np.array([self.get_pct(alea_unc[:,k]) for k in range(50)]).mean()
        
        # avg_alea_unc = alea_unc.mean(dim=1)
        
        if self.params['ignore_below']:
            for j in range(alea_unc.size()[0]):
                num_over_max = (alea_unc[j] < max_unc).sum()
                # if avg_alea_unc[j] < max_unc and (j not in self.ignore_list):
                if num_over_max > num_samples/2 and (j not in self.ignore_list):
                    self.ignore_list.append(j)
        else:
            for j in range(alea_unc.size()[0]):
                num_over_max = (alea_unc[j] > max_unc).sum()
                # if avg_alea_unc[j] > max_unc and (j not in self.ignore_list):
                if num_over_max > num_samples/2 and (j not in self.ignore_list):
                    self.ignore_list.append(j)
        return True
                             
    
    
    def create_dataloaders(self):
        indx = list(range(len(self.dataset)))
        for i in self.ignore_list:
            indx.remove(i)
        sub_ignore = torch.utils.data.Subset(self.dataset,self.ignore_list)
        sub_train = torch.utils.data.Subset(self.dataset,indx)
        
        ignore_loader = torch.utils.data.DataLoader(sub_ignore, self.params['batch_size'], shuffle=True)
        train_loader = torch.utils.data.DataLoader(sub_train, self.params['batch_size'], shuffle=True)
        
        
        
        return train_loader, ignore_loader
     
        
        
        