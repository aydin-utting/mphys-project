import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
#from torchsummary import summary

import numpy as np
import PIL

from models import LeNet, LeNet_KG
from FRDEEP import FRDEEPN
from MiraBest import MiraBest_full, MBFRConfident
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

def output_images_with_class(folder,loader,model):
    
    
    
    model.eval()
    
    try:
        os.makedirs(folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    
    with torch.no_grad():
        for batch, (x,y) in enumerate(loader):
             pred, s = model(x)
             c = pred.argmax(dim=1)
             var = F.softplus(s)
             for i in range(y.size()[0]):
                 imagepath = folder + "/batch{:01d}_img{:01d}_p{}_y{}_var{:.4e}.png".format(batch,i,c[i].item(),y[i].item(),var[i][c[i]].item())
                 plt.imsave(imagepath,x[i][0])


def std_of_alea_func(model,T,dataset):
    all_train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset),shuffle=False)
    model.eval()
    with torch.no_grad():
        
        pp , ss = torch.empty(583, 2, T),torch.empty(583, 2, T)
        cc = torch.empty(583, T)
        for t in range(T):
            print(t)
           
            
            for b, (x,y) in enumerate(all_train_data_loader):
                p, s = model(x)
                c = p.argmax(dim=1)
            pp[:,:,t] = p
            ss[:,:,t] = s
            cc[:,t] = c
        
        
        sstd = torch.empty((583,100))
        for i in range(583):
            for j in range(100):
                sstd[i,j] = torch.sqrt(F.softplus(ss[i,int(cc[i,j].item()),j])).item()
        
        plt.hist2d(sstd.mean(dim=1).numpy(),sstd.std(dim=1).numpy(),bins=[50,50])        
    return pp,ss,cc
        


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


epochs        = 50  # number of training epochs
#valid_size    = 50    # number of samples for validation
batch_size    = 32    # number of samples per mini-batch
imsize        = 50    # image size
num_classes   = 2     # The number of output classes. FRI/FRII
learning_rate = 1e-3  # The speed of convergence
momentum      = 9e-1  # momentum for optimizer
decay         = 0.5e-2  # weight decay for regularisation
random_seed   = 42
step_size = 5
gamma = 0.8
num_bins = 30
prune_milestones = []
prune_sigmas = []
prune_gamma = 2

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def seed_init_fn(x):
   seed = args.seed + x
   np.random.seed(seed)
   random.seed(seed)
   torch.manual_seed(seed)
   return

# Creating the data------------------------------------------------------------

transform = transforms.Compose([
    transforms.CenterCrop(imsize),
    transforms.RandomRotation([0,360],resample=PIL.Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize((0.0229,), (0.0957,))])

train_val_data = MBFRConfident('mirabest', train=True, download=True, transform=transform)


ntrainval = len(train_val_data)

TRAIN_PCT = 0.8
ntrain = int(ntrainval*TRAIN_PCT)

train_data,val_data = torch.utils.data.random_split(train_val_data,[ntrain,ntrainval-ntrain],generator=torch.Generator().manual_seed(random_seed))


#train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)



test_data = MBFRConfident('mirabest', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
ntest  = len(test_data)

#Saved model
outfile = "./mb_lenet_kg_20201110"
IS_TRAINED = True

# Initialise model-----------------------------------------------


def random_prune(pruned_loader, unpruned_data, batch_size):
    length = sum([len(y) for b,(x,y) in enumerate(pruned_loader)])
    pruned_num = len(unpruned_data)-length
    indices = []
    i=0
    while i<length:
        r = random.randrange(len(unpruned_data))
        if r not in indices:
            indices.append(r)
            i+=1
    random_pruned_data = torch.utils.data.Subset(unpruned_data, indices)
    return pruned_num, torch.utils.data.DataLoader(random_pruned_data, batch_size=batch_size, shuffle=True)




# sigmas = [2,3,4,5]
# removed_num = [166,70,47,44]

# df_dict = {}

# for i in range(4):
#     df_dict[str(sigmas[i])+'random']=pd.read_csv(str(sigmas[i])+'sigmas(removed'+str(removed_num[i])+')random.csv' )
#     df_dict[str(sigmas[i])+'notrandom']=pd.read_csv(str(sigmas[i])+'sigmas(removed'+str(removed_num[i])+')notrandom.csv' )
    
    

# params = {'epochs': [100],
#           'batch_size' : [32],
#           'lr' : [1e-2],
#           'decay' : [1e-2,1e-4],
#           'gamma' : [0.98,0.998],
#           'step_size' : [5],
#           'num_bins' : [50],
#           'prune_milestones' : [[10000]],
#           'prune_sigmas' : [[1000]],
#           'prune_gamma' : [50],
#           'ignore_below': [True],
#            'p_i' : [5e-1],
#            'p_f' : [0.99],
#            'T' : [50,100,150,200]
#           }



# prune_param_dict = {'epochs': 3,
#           'batch_size' : 32,
#           'lr' : 1e-3,
#           'decay' : 1e-2,
#           'gamma' : 0.8,
#           'step_size' : 5,
#           'num_bins' : 50,
#           'prune_milestones' : [100],
#           'prune_sigmas' : [2],
#           'prune_gamma' : 1,
#           }

# keys, values = zip(*params.items())
# prune_param_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]


param_dict = {'epochs': 200,
          'batch_size' : 32,
          'lr' : 1e-3,
          'decay' : 1e-2,
          'gamma' : 0.98,
          'step_size' : 5,
          'num_bins' : 50,
          'prune_milestones' : [10000],
          'prune_sigmas' : [0],
          'prune_gamma' : 50,
          'ignore_below' : False,
          'p_i' : 5e-1,
          'p_f' : 0.99,
          'T' : 100
          }

param_dict2 = {'epochs': 100,
          'batch_size' : 32,
          'lr' : 1e-3,
          'decay' : 1e-2,
          'gamma' : 1,
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


model= LeNet_KG(in_chan=1, out_chan=num_classes, imsize=imsize, kernel_size=5)

model,df = train(model,train_data,val_data,param_dict)





# optimizer = torch.optim.Adam(model.parameters(), lr=param_dict['lr'], weight_decay=param_dict['decay'])
# prun = Pruner(train_data, optimizer,params=param_dict)

# tr = prun.sorted_dataloader(model,16,50)

# sort_model= LeNet_KG(in_chan=1, out_chan=num_classes, imsize=imsize, kernel_size=5)


# sort_model,epoch_df,train_df,val_df = standard_train(sort_model,tr,val_data,param_dict2)


# results = param_search(model,params,train_data,val_data,test_data)

# min_params = ast.literal_eval(min(results.keys(),key = lambda k: results[k]['num_errors']))


# fig = plt.figure()
# ax = fig.add_subplot()
# for k in results:
#     ax.plot(results[k]['df']['valloss'])
#     ax.plot(results[k]['df']['trainloss'])
#     break
# fig.show()

# model= LeNet_KG(in_chan=1, out_chan=num_classes, imsize=imsize, kernel_size=5)



# model,df, pruned_loader = train_prune(model, train_data, val_data, param_dict)



# train_loader = torch.utils.data.DataLoader(train_data, batch_size=param_dict['batch_size'], shuffle=True)
# model_without_prune = LeNet_KG(in_chan=1, out_chan=num_classes, imsize=imsize, kernel_size=5)
# model_without_prune, df_without_prune = standard_train(model_without_prune,train_loader,val_data,param_dict)
# df_without_prune.to_csv('No pruning control.csv')
# for p in prune_param_dicts:


#     model_with_prune = LeNet_KG(in_chan=1, out_chan=num_classes, imsize=imsize, kernel_size=5)
#     prune_model = LeNet_KG(in_chan=1, out_chan=num_classes, imsize=imsize, kernel_size=5)
#     model_with_random_prune = LeNet_KG(in_chan=1, out_chan=num_classes, imsize=imsize, kernel_size=5)
    
    
#     prune_model, prune_df, pruned_train_loader = train_prune(prune_model, 
#                                                              train_data, 
#                                                              val_data, 
#                                                              p)
    
#     model_with_prune, df_with_prune = standard_train(model_with_prune,
#                                                      pruned_train_loader,
#                                                      val_data,
#                                                      param_dict)
    
#     pruned_num, random_pruned_train_loader = random_prune(pruned_train_loader,
#                                                           train_data,
#                                                           param_dict['batch_size'])
    
#     model_with_random_prune, df_with_random_prune = standard_train(model_with_random_prune,
#                                                                    random_pruned_train_loader,
#                                                                    val_data,
#                                                                    param_dict)
    
#     df_with_prune.to_csv(str(p['prune_sigmas'][0])+'sigmas(removed:'+str(pruned_num)+'):notrandom.csv')
#     df_with_random_prune.to_csv(str(p['prune_sigmas'][0])+'sigmas(removed:'+str(pruned_num)+'):random.csv')
    
#     fig = plt.figure()
#     ax = fig.add_subplot()
#     ax.plot(df_with_prune['valacc'])
#     ax.plot(df_with_random_prune['valacc'])
#     #ax.plot(df_without_prune['valacc'])
#     ax.legend(['Selective Prune','Random prune'])
#     ax.set_title('Selective vs random vs no prune for removing below top '+str(p['prune_sigmas'][0])+' of data')
#     fig.show()
#     fig.savefig('Selective vs random vs no prune for removing below top '+str(p['prune_sigmas'][0])+' of data.png')
    
# results = param_search(model,params,train_data,val_data,test_data)

# min_params = ast.literal_eval(min(results.keys(),key = lambda k: results[k]['num_errors']))








# if(IS_TRAINED==False):
#     model, df = train(
#         model,
#         train_data,
#         val_data,
#         epochs,  
#         batch_size,   
#         learning_rate,
#         decay,
#         step_size,
#         gamma,
#         num_bins,
#         prune_milestones,
#         prune_sigmas,
#         prune_gamma)
#     torch.save(model.state_dict(), outfile)
    


# else: #if the model is already trained, load the model from save file
#     model.load_state_dict(torch.load(outfile))

# model.eval()


    
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=False)
# with torch.no_grad():
#     test_outputs = []
#     std = []
#     for batch, (x_test,y_test) in enumerate(train_loader):
#           test_pred, test_s = model(x_test)
#           test_c = test_pred.argmax(dim=1)
#           test_std = torch.sqrt(F.softplus(test_s))
             
#     frac_unc = test_std/test_pred
    
#     plt.hist(test_pred.numpy().flatten(),20)
#     plt.hist(test_std.numpy().flatten(),20)
                 

    
# model.eval() #put the model into eval mode so it can be used for analysis




# #output_images_with_class('./testimages',test_loader,model)
# #output_images_with_class('./valimages',val_loader,model)






# train_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=False)
# val_outputs, test_outputs = [], []
# with torch.no_grad():
    
#     for batch, (x_test,y_test) in enumerate(train_loader):
#           test_pred, test_s = model(x_test)
#           test_c = test_pred.argmax(dim=1)
#           test_var = F.softplus(test_s)
#           for i in range(y_test.size()[0]):
#               test_outputs.append([x_test[i][0],test_c[i].item(),y_test[i].item(),test_var[i][test_c[i]].item()])

# sorted_test_output = sorted(test_outputs,key = lambda i: i[3]) 

# s=0
# for it in sorted_test_output:
#     if it[1]==it[2]:
#         s+=1
        
# acc = s/len(sorted_test_output)

# print("test acc: ", acc)


