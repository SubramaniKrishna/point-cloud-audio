# Evaluating the trained point cloud 
# 1. Varying the sampling rate
# 2. Varying N/H
# 3. (Potentially) Reassigning the spectrogram
# 4. Keeping the top k points (in magnitude or some other relevance measure)

import matplotlib.pyplot as pyp
import numpy as np
import torch
np.random.seed(1)
torch.manual_seed(1)
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import csv
from scipy.signal import butter,sosfilt, decimate
import librosa
from tqdm import trange
from tqdm import tqdm
import json
import glob
import yaml
import torch
import torch.nn as nn
from torchinfo import summary
from dataset import *
from data_processing import *

import sys
sys.path.append('../set_transformer-master/')
sys.path.append('../DeepSets-master/PointClouds/')


from data_modelnet40 import ModelFetcher
from modules import ISAB, PMA, SAB
import classifier
import modelnet
from prettytable import PrettyTable
from thop import profile, clever_format

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Point cloud max-K subsampling
def pc_maxK(x,farr,Kmax):
    subsampled_x = []
    subsampled_x_fs = []
    for i in range(x.shape[1]):
        indices = (-x[:,i]).argsort()[:Kmax]
        xthresh_s = x[:,i][indices]
        fthresh = farr[indices]
        subsampled_x.append(xthresh_s.reshape(1,xthresh_s.shape[0]))
        subsampled_x_fs.append(fthresh.reshape(1,fthresh.shape[0]))

    subsampled_x = np.concatenate(subsampled_x,axis = 0).T
    subsampled_x_fs = np.concatenate(subsampled_x_fs,axis = 0).T

    return subsampled_x,subsampled_x_fs

def pc_randK(x,farr,Kmax):
    subsampled_x = []
    subsampled_x_fs = []
    for i in range(x.shape[1]):
        indices = np.random.permutation(x[:,i].shape[0])[:Kmax]
        xthresh_s = x[:,i][indices]
        fthresh = farr[indices]
        subsampled_x.append(xthresh_s.reshape(1,xthresh_s.shape[0]))
        subsampled_x_fs.append(fthresh.reshape(1,fthresh.shape[0]))

    subsampled_x = np.concatenate(subsampled_x,axis = 0).T
    subsampled_x_fs = np.concatenate(subsampled_x_fs,axis = 0).T

    return subsampled_x,subsampled_x_fs


# Loading the network and weights
# 1: settransformer, 2: deepsets, 3: gcnn
net_c = 1

run_id = 'run-20210414_235515-1bsbgs4l'
# Point to directory of wandb parameters (config.yaml)
pth_yml = glob.glob('./wandb/' + run_id + '/files/*.yaml')[0]
pth_pth = glob.glob('./wandb/' + run_id + '/files/*.pth')[0]
with open(pth_yml) as f:
    dict_params = yaml.safe_load(f)

Nfft = dict_params['window_size']['value']
fsog = dict_params['sampling_rate']['value']
hf = dict_params['hop_factor']['value']
tDb = dict_params['trim_dB']['value']
batch_size = 4

if(net_c == 1):
    class SetTransformer(nn.Module):
        def __init__(
            self,
            dim_input=2,
            num_outputs=1,
            dim_output=10,
            num_inds=4,
            dim_hidden=4,
            num_heads=2,
            ln=False,
        ):
            super(SetTransformer, self).__init__()
            self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
            )
            self.dec = nn.Sequential(
                # nn.Dropout(),
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                # nn.Dropout(),
                nn.Linear(dim_hidden, dim_output),
            )

        def forward(self, X):
            return self.dec(self.enc(X)).squeeze()

    dhidden = dict_params['dhidden']['value']
    nheads = dict_params['nheads']['value']
    ninds = dict_params['ninds']['value']
    # Training NN
    # pth file location
    model = SetTransformer(dim_hidden=dhidden, num_heads=nheads, num_inds=ninds).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(pth_pth,map_location = device))

elif(net_c == 2):
    network_dim = 512
    model = classifier.DTanh(d_dim = network_dim,x_dim = 2, pool='max1')
    # Training NN
    # pth file location
    dir_pth = './deepsetsparams/settransformerpc(2021-02-11 14:28:29.972221)_net.pth'
    model.load_state_dict(torch.load(dir_pth,map_location = device))

count_parameters(model)
# criterion = F.cross_entropy
#model = nn.DataParallel(model)
model = model.to(device)
# model.eval()

# Load ESC-50 filenames into list
list_audio_locs,l = load_esc()
nclass = max(l) + 1
list_train,l_train,list_test,l_test = tt_split(list_audio_locs,l,f = 0.8)

# Varying N,Fs

# Compute STFT and load data into frames
fsog = dict_params['sampling_rate']['value']
# list_Fs = [fsog,0.95*fsog,0.9*fsog,0.8*fsog,0.5*fsog,0.1*fsog]
list_Fs = [fsog,0.75*fsog,0.5*fsog,0.25*fsog]
list_N = [2*Nfft,(int)(1.5*Nfft),(int)(1.25*Nfft),(int)(1.05*Nfft),Nfft,(int)(0.95*Nfft),(int)(0.9*Nfft),(int)(0.8*Nfft),(int)(0.7*Nfft),(int)(0.6*Nfft),(int)(0.5*Nfft),(int)(0.25*Nfft),(int)(0.1*Nfft)]
# list_Fs = [fsog,0.95*fsog]
# list_N = [2*Nfft,(int)(1.5*Nfft)]
# Nfft = 4096
# hf = 0.5

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
        print(N,F)

        d_esc = []
        l_esc = []
        for i in tqdm(range(len(list_test))):
            fi = list_test[i]
            x, fsog = librosa.load(fi,sr = fsog)
            x, index = librosa.effects.trim(x,top_db = tDb)
            x = librosa.resample(x, fsog, fs, res_type='kaiser_fast',scale= True)
            x = librosa.stft(x,n_fft = (int)(2**(np.ceil(np.log2(Nfft)))),win_length = Nfft, hop_length=(int)(Nfft*hf), window = 'hann')/Nfft
            a = np.log(1.0e-8 + np.abs(x))
            # a = np.random.randn(*a.shape)
            d_esc.append(a)
            l_esc.append(l_test[i]*np.ones(a.shape[1]))
        x_test = np.concatenate(d_esc, axis=1)
        y_test = np.concatenate(l_esc, axis=0).astype(int)
        farr = np.linspace(0,fs/2,x_test.shape[0])/fs

        test_dataset = ESC_pc(x = x_test,y = y_test, farr = farr)
        data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)

        criterion = nn.CrossEntropyLoss()
        losses, total, correct = [], 0, 0
        for imgs,lbls in data_loader_test:
            if(lbls.shape[0] < 2):
                continue
            # print(imgs.shape)
            # imgs = torch.Tensor(imgs).cuda()
            # lbls = torch.Tensor(lbls).long().cuda()
            imgs = imgs.cuda()
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
        # list_NF.append([N,F])
        # list_finval.append([avg_loss,avg_acc])
        print(f"Test acc {avg_acc:.3f}")
        dict_errs["data"][F].append(avg_acc)
        del total, correct, avg_loss,avg_acc
json_file = './plots/pc_framewise_settransformer_' + run_id + '_data.json'
with open(json_file, 'w') as outfile:
    json.dump(dict_errs, outfile)
# Plottings
# pyp.figure()
# pyp.subplot(211)
# pyp.title("Accuracy vs N")
# for idx,kv in enumerate(list_NF):
#     pyp.plot(kv[0],list_finval[idx][1],'b.')
# pyp.subplot(212)
# pyp.title("Accuracy vs Fs")
# for idx,kv in enumerate(list_NF):
#     pyp.plot(kv[1],list_finval[idx][1],'b.')
# pyp.tight_layout()
# pyp.show()
# pyp.savefig('./NfsErrors.png')
# pyp.close()



# Varying K (number of max points to keep from spectrum)
"""
criterion = nn.CrossEntropyLoss()
Nfft = dict_params['window_size']['value']
fsog = dict_params['sampling_rate']['value']
list_K = np.arange(1,Nfft//2,50)
list_K[-1] = Nfft//2
Nruns = 10

dict_errs_randK = {"data":{(int)(k):0 for k in list_K}}
dict_errs_maxK = {"data":{(int)(k):0 for k in list_K}}

dict_errs_maxK["list_K"] = list_K.tolist()
dict_errs_randK["list_K"] = list_K.tolist()

d_esc = []
l_esc = []
for i in tqdm(range(len(list_test))):
    fi = list_test[i]
    x, fsog = librosa.load(fi,sr = fsog)
    x, index = librosa.effects.trim(x,top_db = tDb)
    x = librosa.stft(x,n_fft = Nfft,win_length = Nfft, hop_length=(int)(Nfft*hf), window = 'hann')/Nfft
    a = np.log(1.0e-8 + np.abs(x))
    # a = np.random.randn(*a.shape)
    d_esc.append(a)
    l_esc.append(l_test[i]*np.ones(a.shape[1]))
x_test = np.concatenate(d_esc, axis=1)
y_test = np.concatenate(l_esc, axis=0).astype(int)
farr = np.linspace(0,fsog/2,x_test.shape[0])/fsog

for Km in list_K:
    # Random K sampling
    list_finval = []
    for i in range(Nruns):
        # subsampling
        xss,farrss = pc_randK(x_test,farr,Km)
        test_dataset = ESC_pc_ss(x = xss,y = y_test, farr = farrss)
        data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)

        losses, total, correct = [], 0, 0
        for imgs,lbls in data_loader_test:
            if(lbls.shape[0] < batch_size):
                continue
            # print(imgs.shape)
            # imgs = torch.Tensor(imgs).cuda()
            # lbls = torch.Tensor(lbls).long().cuda()
            # print(imgs.shape)
            imgs = imgs.cuda()
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
        xss,farrss = pc_maxK(x_test,farr,Km)
        test_dataset = ESC_pc_ss(x = xss,y = y_test, farr = farrss)
        data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)

        criterion = nn.CrossEntropyLoss()
        losses, total, correct = [], 0, 0
        for imgs,lbls in data_loader_test:
            if(lbls.shape[0] < batch_size):
                continue
            # print(imgs.shape)
            # imgs = torch.Tensor(imgs).cuda()
            # lbls = torch.Tensor(lbls).long().cuda()
            imgs = imgs.cuda()
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

json_file = './plots/pc_framewise_settransformer_randKSS_' + run_id + '_data.json'
with open(json_file, 'w') as outfile:
    json.dump(dict_errs_randK, outfile)

json_file = './plots/pc_framewise_settransformer_maxKSS_' + run_id + '_data.json'
with open(json_file, 'w') as outfile:
    json.dump(dict_errs_maxK, outfile)
# Plottings
# pyp.figure()
# pyp.title("Accuracy vs Number of Points Kept")
# pyp.plot(list_K,list_finval,'b.')
# pyp.show()
# pyp.savefig('./accK.png')
# pyp.close()
"""
