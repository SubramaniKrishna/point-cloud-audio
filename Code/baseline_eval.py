import numpy as np
np.random.seed(0)
import scipy as sp
import matplotlib.pyplot as pyp
import librosa
from tqdm import trange
from tqdm import tqdm

from datetime import datetime
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import csv

import torch
torch.manual_seed(0)
import torch.nn as nn
from torchinfo import summary
from dataset import *
from data_processing import *
import json
import glob
import yaml
from prettytable import PrettyTable

# Counting Model Parameters
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
# Subsampling Functions (preserving size of vector, basically zeroing out the entries)

"""
Defining the Network
"""

class baseline_ff(nn.Module):
    """
    Feedforward baseline classifier

    Constructor Parameters
    ----------------------
    layer_dims : list of integers
        List containing the dimensions of consecutive layers(Including input layer{dim(X)} and excluding latent layer)
    """
    # Defining the constructor to initialize the network layers and activations
    def __init__(self, layer_dims,nclasses,p = 0.5):
        super().__init__()
        self.layer_dims = layer_dims

        # Dropout to emulate rand sampling
        self.dpout = nn.Dropout(p=p)

        # Initializing the Model as a torch sequential model
        self.ENC_NN = nn.Sequential()

        # Currently using Linear layers with ReLU activations(potential hyperparams)
        # This Loop defines the layers just before the latent space (input(X) -> layer[0] -> layer[1] .... -> layer[n - 1])
        for i, (in_size, out_size) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            self.ENC_NN.add_module(name = "Encoder_Layer_{:d}".format(i), module = nn.Linear(in_size, out_size))
            # self.ENC_NN.add_module(name = "BatchNorm{:d}".format(i), module = nn.BatchNorm1d(out_size))
            self.ENC_NN.add_module(name = "Activation_{:d}".format(i), module = nn.LeakyReLU())
            
            

        # Convert Layer to the Intermediary code
        # self.ENC_NN.add_module(name = "BatchNormFinal", module = nn.BatchNorm1d(layer_dims[:-1]))
        self.ENC_NN.add_module(name = "Code_Linear", module = nn.Linear(layer_dims[-1], nclasses))
        # self.ENC_NN.add_module(name = "BatchNormFinal", module = nn.BatchNorm1d(nclasses))
        self.ENC_NN.add_module(name = "Softmax", module = nn.Softmax())

    def forward(self, x):
        """
        Forward pass of the input to obtain logits
        Inputs
        ------
        x : torch.tensor
            Input tensor
        """
        # Forward pass till the n-1'th layer
        y = self.ENC_NN(self.dpout(x))

        return y

# Funtions to replace the non-max/non-random chosen elements (as an equivalent experiment to the point cloud sub-sampling experiment)
# Replacement to ensure size consistentcy with baseline NN
def pc_maxK_replace(x,Kmax):
    xreplace = []
    for i in range(x.shape[1]):
        temp = np.zeros(x[:,i].shape[0])
        indices = (-x[:,i]).argsort()[:Kmax]
        temp[indices] = x[:,i][indices]
        xreplace.append(temp)

    xreplace = np.array(xreplace).T
    return xreplace

def pc_randK_replace(x,Kmax):
    xreplace = []
    for i in range(x.shape[1]):
        temp = np.zeros(x[:,i].shape[0])
        indices = np.random.permutation(x[:,i].shape[0])[:Kmax]
        temp[indices] = x[:,i][indices]
        xreplace.append(temp)

    xreplace = np.array(xreplace).T
    return xreplace


run_id = 'run-20210413_235750-2m4htfga'
# Point to directory of wandb parameters (config.yaml)
pth_yml = glob.glob('./wandb/' + run_id + '/files/*.yaml')[0]
pth_pth = glob.glob('./wandb/' + run_id + '/files/*.pth')[0]
with open(pth_yml) as f:
    dict_params = yaml.safe_load(f)

Nfft = dict_params['window_size']['value']
fsog = dict_params['sampling_rate']['value']
hf = dict_params['hop_factor']['value']
tDb = dict_params['trim_dB']['value']
layers = dict_params['layers']['value']
batch_size = dict_params['batch_size']['value']
dropout_prob = dict_params['dropout_prob']['value']

# Load ESC-50 filenames into list
list_audio_locs,l = load_esc()
nclass = max(l) + 1
list_train,l_train,list_test,l_test = tt_split(list_audio_locs,l,f = 0.8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = baseline_ff(layer_dims = layers, nclasses = nclass,p = dropout_prob).to(device)
model.load_state_dict(torch.load(pth_pth,map_location = device))
model.eval()
count_parameters(model)

# Varying N,Fs

# list_Fs = [fsog,0.95*fsog,0.9*fsog,0.8*fsog,0.5*fsog,0.1*fsog]
list_Fs = [fsog,0.75*fsog,0.5*fsog,0.25*fsog]
list_N = [Nfft,(int)(0.95*Nfft),(int)(0.9*Nfft),(int)(0.8*Nfft),(int)(0.7*Nfft),(int)(0.6*Nfft),(int)(0.5*Nfft),(int)(0.25*Nfft),(int)(0.1*Nfft)]
Nfftog = Nfft
# list_Fs = [fsog,0.95*fsog]
# list_N = [2*Nfft,(int)(1.5*Nfft)]
# Nfft = 4096
# hf = 0.5

# Varying N,Fs
# list_NF = []
# list_finval = []
dict_errs = {"data":{k:[] for k in list_Fs}}
dict_errs["list_Fs"] = list_Fs
dict_errs["list_N"] = list_N

for F in list_Fs:
    for N in list_N:
        Nfft = N
        fs = F
        d_esc = []
        l_esc = []
        print(F,N)

        # Test
        d_esc = []
        l_esc = []
        for i in tqdm(range(len(list_test))):
            fi = list_test[i]
            x, fsog = librosa.load(fi,sr = fsog)
            x, index = librosa.effects.trim(x,top_db = tDb)
            x = librosa.resample(x, fsog,fs , res_type='kaiser_fast',scale= True)
            x = librosa.stft(x,n_fft = Nfftog, win_length = Nfft, hop_length=(int)(Nfft*hf), window = 'hann')/Nfftog
            a = np.log(1.0e-8 + np.abs(x))
            # a = np.random.randn(*a.shape)
            d_esc.append(a)
            l_esc.append(l_test[i]*np.ones(a.shape[1]))
        x_test = np.concatenate(d_esc, axis=1)
        y_test = np.concatenate(l_esc, axis=0).astype(int)

        test_dataset = ESC_baseline(x = x_test,y = y_test)
        data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)

        losses, total, correct = [], 0, 0
        for iteration, (yi,xi) in enumerate(data_loader_test):
            yi, xi = yi.to(device, non_blocking=True), xi.to(device, non_blocking=True)
            # print(xi.shape,yi.shape)
            pi = model(xi.float())
            # print(pi.shape,yi.shape)
            total += yi.shape[0]
            correct += (pi.argmax(dim=1) == yi).sum().item()
        avg_acc = correct / total

        # dict_testN[N] = [avg_loss,avg_acc]
        # dict_testF[F] = [avg_loss,avg_acc]
        # list_NF.append([N,F])
        # list_finval.append([avg_loss,avg_acc])
        print(f"Test acc {avg_acc:.3f}")
        dict_errs["data"][F].append(avg_acc)
        del total, correct,avg_acc
json_file = './plots/baseline_framewise_' + run_id + '_data.json'
with open(json_file, 'w') as outfile:
    json.dump(dict_errs, outfile)


# Sub-sampling with zero replacement
"""
Nfft = dict_params['window_size']['value']
fsog = dict_params['sampling_rate']['value']
batch_size = 2
list_K = np.arange(10,Nfft//2,100)
# list_K = [Nfft//2 + 1]
list_K[-1] = Nfft//2 + 1
Nruns = 10

dict_errs_randK = {"data":{(int)(k):0 for k in list_K}}
dict_errs_maxK = {"data":{(int)(k):0 for k in list_K}}

dict_errs_maxK["list_K"] = list_K.tolist()
dict_errs_randK["list_K"] = list_K.tolist()

d_esc = []
l_esc = []
for i in tqdm(range(len(list_test))):
    fi = list_test[i]
    x, fs = librosa.load(fi,sr = 44100)
    x, index = librosa.effects.trim(x,top_db = tDb)
    x = librosa.stft(x,n_fft = Nfft, win_length = Nfft, hop_length=(int)(Nfft*hf), window = 'hann')/Nfft
    a = np.log(1.0e-8 + np.abs(x))
    # a = np.random.randn(*a.shape)
    d_esc.append(a)
    l_esc.append(l_test[i]*np.ones(a.shape[1]))
x_test = np.concatenate(d_esc, axis=1)
y_test = np.concatenate(l_esc, axis=0).astype(int)


for Km in list_K:
    # Random K sampling
    list_finval = []
    for i in range(Nruns):
        # subsampling
        xss = pc_randK_replace(x_test,Km)
        test_dataset = ESC_baseline(x = xss,y = y_test)
        data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)
        
        criterion = nn.CrossEntropyLoss()
        losses, total, correct = [], 0, 0
        for lbls,imgs in data_loader_test:
            # if(lbls.shape[0] < 10):
                # continue
            # print(imgs.shape)
            # imgs = torch.Tensor(imgs).cuda()
            # lbls = torch.Tensor(lbls).long().cuda()
            imgs = imgs.float().cuda()
            lbls = lbls.cuda()
            preds = model(imgs)
            # print(preds.shape,lbls.shape)
            # loss = criterion(preds, lbls)

            # losses.append(loss.item())
            
            total += lbls.shape[0]
            # print(preds)
            # print(lbls.shape,imgs.shape)
            # print(preds.shape)
            # print(preds.argmax(dim = 1))
            #print(preds.shape,preds.argmax(dim = 1).shape,preds.argmax(dim = 1),lbls.shape)
            correct += (preds.argmax(dim=1) == lbls).sum().item()

        avg_loss, avg_acc = np.mean(losses), correct / total

        # dict_testN[N] = [avg_loss,avg_acc]
        # dict_testF[F] = [avg_loss,avg_acc]
        list_finval.append(avg_acc)
        print("RandK, ", Km)
        print(f"Test acc {avg_acc:.3f}")
        del total, correct, avg_loss,avg_acc
    list_finval = np.array(list_finval)
    dict_errs_randK["data"][(int)(Km)] = [np.mean(list_finval),np.var(list_finval)]

    # Max K sampling
    list_finval = []
    for i in range(Nruns):
        # subsampling
        xss = pc_maxK_replace(x_test,Km)
        test_dataset = ESC_baseline(x = xss,y = y_test)
        data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)

        criterion = nn.CrossEntropyLoss()
        losses, total, correct = [], 0, 0
        for lbls,imgs in data_loader_test:
            # if(lbls.shape[0] < 10):
                # continue
            # print(imgs.shape)
            # imgs = torch.Tensor(imgs).cuda()
            # lbls = torch.Tensor(lbls).long().cuda()
            imgs = imgs.float().cuda()
            lbls = lbls.cuda()
            preds = model(imgs)
            # print(preds.shape,lbls.shape)
            # loss = criterion(preds, lbls)

            # losses.append(loss.item())
            
            total += lbls.shape[0]
            # print(preds)
            # print(lbls.shape,imgs.shape)
            # print(preds.shape)
            # print(preds.argmax(dim = 1))
            #print(preds.shape,preds.argmax(dim = 1).shape,preds.argmax(dim = 1),lbls.shape)
            correct += (preds.argmax(dim=1) == lbls).sum().item()

        avg_loss, avg_acc = np.mean(losses), correct / total

        # dict_testN[N] = [avg_loss,avg_acc]
        # dict_testF[F] = [avg_loss,avg_acc]
        list_finval.append(avg_acc)
        print("maxK, ", Km)
        print(f"Test acc {avg_acc:.3f}")
        del total, correct, avg_loss,avg_acc
    list_finval = np.array(list_finval)
    dict_errs_maxK["data"][(int)(Km)] = [np.mean(list_finval),np.var(list_finval)]

json_file = './plots/baseline_framewise_randKSS_' + run_id + '_data.json'
with open(json_file, 'w') as outfile:
    json.dump(dict_errs_randK, outfile)

json_file = './plots/baseline_framewise_maxKSS_' + run_id + '_data.json'
with open(json_file, 'w') as outfile:
    json.dump(dict_errs_maxK, outfile)
"""