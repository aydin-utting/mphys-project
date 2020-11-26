
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

from pruner import Pruner

import pandas as pd


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





def train(model, train_data, val_data, params):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['decay'])
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=params['p_i'], dampening=params['p_i'])
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])
    
    pruner = Pruner(train_data,optimizer ,params)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=params['batch_size'], shuffle=True)
    # output_file = 'bs_'+str(batch_size)+'_lr_'+str(learning_rate)+'_wd_'+str(decay)+'.csv'
    # with open(output_file, 'w', newline="") as f_out:
    #     writer = csv.writer(f_out, delimiter=',')
    #     writer.writerow(["Epoch",'Train Loss','Val Loss','Val Acc'])    
    
    
    df = pd.DataFrame(index=list(range(params['epochs'])),columns = ["trainacc","trainloss","valacc","valloss"])
    
    # fig = plt.gcf()
    # fig.show()
    # fig.canvas.draw()
    
    for epoch in range(params['epochs']):  # loop over the dataset multiple times
        t1 = time.perf_counter()
        model.train()
        train_losses,  train_accs = [], []; acc = 0
        for batch, (x_train, y_train) in enumerate(train_loader):
            
            model.zero_grad()
            pred, std = model(x_train)
            
            if epoch>params['prune_milestones'][0]:
                loss = F.cross_entropy(pred,y_train)
            else:
                d, yd = model.sample_sigma(pred, std,y_train)
                loss = F.cross_entropy(pred,y_train) + loss_l2(d,yd)
            
            
            loss.backward()
            optimizer.step()
            
            # optimizer.param_groups[0]['lr'] = get_lr(epoch, params['lr'], params['gamma'])
            # optimizer.param_groups[0]['momentum'] = get_momentum(epoch, params['p_i'], params['p_f'], params['T'])
            # optimizer.param_groups[0]['dampening'] = get_momentum(epoch, params['p_i'], params['p_f'], params['T'])
            
            
            acc = (pred.argmax(dim=-1) == y_train).to(torch.float32).mean()
            train_accs.append(acc.mean().item())
            train_losses.append(loss.item())   
            
        with torch.no_grad():
            model.eval()
            val_losses, val_accs = [], []; acc = 0
            for i, (x_val, y_val) in enumerate(val_loader):
                val_pred, val_std = model(x_val)
                
                if epoch>params['prune_milestones'][0]:
                    val_loss = F.cross_entropy(val_pred,y_val)
                else:
                    val_d, val_yd = model.sample_sigma(val_pred, val_std,y_val)
                    val_loss =  F.cross_entropy(val_pred,y_val) + loss_l2(val_d,val_yd)
                
                # val_d, val_yd = model.sample_sigma(val_pred, val_std,y_val)
                # loss =  F.cross_entropy(val_pred,y_val) + loss_l2(val_d,val_yd)
                acc = (val_pred.argmax(dim=-1) == y_val).to(torch.float32).mean()
                val_losses.append(val_loss.item())
                val_accs.append(acc.mean().item())
        
            train_loader  = pruner.prune_dataloaders(model, train_loader)
        
        scheduler.step()
            
        
        print('Decay: {}, Epoch: {}, Loss: {}, VAccuracy: {}, Ignore_list_Size: {}, LR: {}'.format(params['decay'],epoch, np.mean(val_losses), np.mean(val_accs), len(pruner.ignore_list),optimizer.param_groups[0]['lr']))
        
        df.loc[epoch, 'trainacc'] = np.mean(train_accs)
        df.loc[epoch,'trainloss'] = np.mean(train_losses)
        df.loc[epoch,'valacc'] = np.mean(val_accs)
        df.loc[epoch,'valloss'] = np.mean(val_losses)
                                     
        # plt.plot(df[:epoch+1]['valloss'])
        # plt.plot(df[:epoch+1]['trainloss'])
        
        # plt.pause(0.01)  # I ain't needed!!!
        # fig.canvas.draw()
        
        
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.plot(df['valloss'])
    ax1.plot(df['trainloss'])
    ax1.grid()
    ax1.set_xlabel('Epoch')
    fig1.show()
    return model, df


def train_prune(model, train_data, val_data, params):
    
    init_model_dict = model.state_dict()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])
    
    pruner = Pruner(train_data,optimizer ,params)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=params['batch_size'], shuffle=True)
    # output_file = 'bs_'+str(batch_size)+'_lr_'+str(learning_rate)+'_wd_'+str(decay)+'.csv'
    # with open(output_file, 'w', newline="") as f_out:
    #     writer = csv.writer(f_out, delimiter=',')
    #     writer.writerow(["Epoch",'Train Loss','Val Loss','Val Acc'])    
    
    
    
    df = pd.DataFrame(index=list(range(params['epochs'])),columns = ["trainacc","trainloss","valacc","valloss"])
    
    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()

    
    epoch_trainaccs, epoch_valaccs = [], []
    epoch_trainloss, epoch_valloss = [], []
    for epoch in range(params['epochs']):  # loop over the dataset multiple times
        t1 = time.perf_counter()
        model.train()
        train_losses,  train_accs = [], []; acc = 0
        for batch, (x_train, y_train) in enumerate(train_loader):
            
            model.zero_grad()
            pred, std = model(x_train)
            
            if (epoch>params['prune_milestones'][0]):
                d, yd = model.sample_sigma(pred, std,y_train)
                loss = F.cross_entropy(pred,y_train) + loss_l2(d,yd)
            else:
                loss = F.cross_entropy(pred,y_train)
            
            loss.backward()
            optimizer.step()
            
            acc = (pred.argmax(dim=-1) == y_train).to(torch.float32).mean()
            train_accs.append(acc.mean().item())
            train_losses.append(loss.item())   
            
        with torch.no_grad():
            model.eval()
            val_losses, val_accs = [], []; acc = 0
            for i, (x_val, y_val) in enumerate(val_loader):
                val_pred, val_std = model(x_val)
                val_d, val_yd = model.sample_sigma(val_pred, val_std,y_val)
                loss =  F.cross_entropy(val_pred,y_val) + loss_l2(val_d,val_yd)
                acc = (val_pred.argmax(dim=-1) == y_val).to(torch.float32).mean()
                val_losses.append(loss.item())
                val_accs.append(acc.mean().item())
        
            train_loader  = pruner.prune_dataloaders(model, train_loader)
        
        scheduler.step()
            
        
        print('Decay: {}, Epoch: {}, Loss: {}, TAccuracy: {}, Ignore_list_Size: {}, LR: {}'.format(params['decay'],epoch, np.mean(val_losses), np.mean(train_accs), len(pruner.ignore_list),optimizer.param_groups[0]['lr']))
        
        df.loc[epoch, 'trainacc'] = np.mean(train_accs)
        df.loc[epoch,'trainloss'] = np.mean(train_losses)
        df.loc[epoch,'valacc'] = np.mean(val_accs)
        df.loc[epoch,'valloss'] = np.mean(val_losses)
                 
        plt.plot(df[:epoch+1]['valloss'])
        plt.plot(df[:epoch+1]['trainloss'])
        
        plt.pause(0.01)  # I ain't needed!!!
        fig.canvas.draw()
        
        epoch_trainaccs.append(np.mean(train_accs))
        epoch_valaccs.append(np.mean(val_accs))
        epoch_trainloss.append(np.mean(train_losses))
        epoch_valloss.append(np.mean(val_losses))
        _results = [time.perf_counter()-t1,epoch, np.mean(train_losses), np.mean(val_losses), np.mean(val_accs)]
        # with open(output_file, 'a', newline="") as f_out:
        #     writer = csv.writer(f_out, delimiter=',')
        #     writer.writerow(_results)
        
    # with torch.no_grad():
    #     unc = []
    #     model.eval()
    #     for  b,(x,y) in enumerate(train_loader):
    #         pred,s = model(x)
    #         c = pred.argmax(dim=1)
    #         for i in range(len(c)):
    #             unc.append(torch.sqrt(F.softplus(s[i,c[i]])).item())
    #     fig = plt.figure()
    #     ax = fig.add_subplot()
    #     ax.hist(unc,100)
    #     ax.set_title('Aleatoric Uncertainty distribution of training data after pruning')
        
        
        
        
    return model, df, train_loader

def standard_train(model,train_loader,val_data,params):

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=params['batch_size'], shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['lr'])
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=params['p_i'], dampening=params['p_i'])
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])
    
    df = pd.DataFrame(index=list(range(params['epochs'])),columns = ["trainacc","trainloss","valacc","valloss"])
    
    
    train_batch_num = [b for b,d in enumerate(train_loader)][-1]+1
    val_batch_num = [b for b,d in enumerate(val_loader)][-1]+1
    
    train_df = pd.DataFrame(columns = ["trainacc","trainloss"])
    val_df = pd.DataFrame(columns = ['valacc',"valloss"])
    
    for epoch in range(params['epochs']):  # loop over the dataset multiple times
    
        model.train()
        train_losses, train_accs = [], []; acc = 0
        for batch, (x_train, y_train) in enumerate(train_loader):
    
            model.zero_grad()
    
            pred, std = model(x_train)
            # add line in here to call sample_sigma()
            loss = F.cross_entropy(pred, y_train)
            loss.backward()
            
            # optimizer.param_groups[0]['lr'] = get_lr(epoch, params['lr'], params['gamma'])
            # optimizer.param_groups[0]['momentum'] = get_momentum(epoch, params['p_i'], params['p_f'], params['T'])
            # optimizer.param_groups[0]['dampening'] = get_momentum(epoch, params['p_i'], params['p_f'], params['T'])
            
            
            optimizer.step()
            
            acc = (pred.argmax(dim=-1) == y_train).to(torch.float32).mean()
            train_accs.append(acc.mean().item())
            train_losses.append(loss.item())
            
            train_df.loc[epoch*train_batch_num+batch,'trainloss'] = loss.item()
            train_df.loc[epoch*train_batch_num+batch,'trainacc'] = acc.mean().item()
            
            
            
    
        with torch.no_grad():
            model.eval()
            val_losses, val_accs = [], []; acc = 0
            for i, (x_val, y_val) in enumerate(val_loader):
                val_pred, val_std = model(x_val)
                loss = F.cross_entropy(val_pred, y_val)
                acc = (val_pred.argmax(dim=-1) == y_val).to(torch.float32).mean()
                val_losses.append(loss.item())
                val_accs.append(acc.mean().item())
                val_df.loc[epoch*val_batch_num+i,'valloss'] = loss.item()
                val_df.loc[epoch*val_batch_num+i,'valacc'] = acc.mean().item()
    
        print('Epoch: {}, Loss: {}, Accuracy: {}'.format(epoch, np.mean(val_losses), np.mean(val_accs)))
        df.loc[epoch, 'trainacc'] = np.mean(train_accs)
        df.loc[epoch,'trainloss'] = np.mean(train_losses)
        df.loc[epoch,'valacc'] = np.mean(val_accs)
        df.loc[epoch,'valloss'] = np.mean(val_losses)
        
        scheduler.step()
        
    print("Final val error: ",100.*(1 - acc))

    return model, df, train_df, val_df
