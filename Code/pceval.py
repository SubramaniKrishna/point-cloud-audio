"""
# Experiments with the trained Model (Framewise Settransformer FST)
1. Varying the sampling rate
2. Varying N/H
3. Keeping the top k points (in magnitude or some other relevance measure)
"""

import numpy as np
import torch
import torch.nn as nn
import json
import librosa
from tqdm import trange
from tqdm import tqdm
from dataset import *
from data_processing import *
from utils import *
from models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Point to the network.pth file and the .json config dict
pth_json = './model_saves/FST(2021-04-26 21:49:40.977943)_config.json'
pth_pth = './model_saves/FST(2021-04-26 21:49:40.977943)_net.pth'
# Load config file
with open(pth_json) as json_file:
    dict_params = json.load(json_file)

# Set seeds for reproducability (and to ensure correct train/test split)
np_seed = dict_params["numpy_seed"]
torch_seed = dict_params["torch_seed"]
np.random.seed(np_seed)
torch.manual_seed(torch_seed)

# Load audio processing and model related parameters
Nfft = dict_params['window_size']
fsog = dict_params['sampling_rate']
hf = dict_params['hop_factor']
tDb = dict_params['trim_dB']
batch_size = 8

dhidden = dict_params['dhidden']
nheads = dict_params['nheads']
ninds = dict_params['ninds']
model = ST(dim_hidden=dhidden, num_heads=nheads, num_inds=ninds).to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(pth_pth,map_location = device))

# Load ESC-50 filenames into list
list_audio_locs,l = load_esc(loc = '../../../../Code/Experiments/GNN/ESC-50-master/meta/esc50.csv',loc_audio = '../../../../Code/Experiments/GNN/ESC-50-master/audio/')
nclass = max(l) + 1
list_train,l_train,list_test,l_test = tt_split(list_audio_locs,l,f = 0.8)

# Experiment 1: Accuracy by varying N,Fs
list_Fs = [fsog,32000,0.5*fsog,0.25*fsog]
list_N = [2*Nfft,(int)(1.5*Nfft),(int)(1.25*Nfft),(int)(1.05*Nfft),Nfft,(int)(0.95*Nfft),(int)(0.9*Nfft),(int)(0.8*Nfft),(int)(0.7*Nfft),(int)(0.6*Nfft),(int)(0.5*Nfft),(int)(0.25*Nfft),(int)(0.1*Nfft)]
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
            if(lbls.shape[0] < batch_size):
                continue
            imgs = imgs.cuda()
            lbls = lbls.cuda()
            preds = model(imgs)
            total += lbls.shape[0]
            correct += (preds.argmax(dim=1) == lbls).sum().item()

        avg_loss, avg_acc = np.mean(losses), correct / total
        print(f"Test acc {avg_acc:.3f}")
        dict_errs["data"][F].append(avg_acc)
        del total, correct, avg_loss,avg_acc
# Saving accuracy data to .json
json_file = './paper_plots/FST_expt1.json'
with open(json_file, 'w') as outfile:
    json.dump(dict_errs, outfile)

# Experiment 2: Point Cloud Subsampling (top K and Random K)
Nfft = dict_params['window_size']
fsog = dict_params['sampling_rate']
# list of K points (< Size of PC)
list_K = np.arange(1,Nfft//2,50)
list_K[-1] = Nfft//2
# Number of runs to average over (For the Random Sampling)
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
    d_esc.append(a)
    l_esc.append(l_test[i]*np.ones(a.shape[1]))
x_test = np.concatenate(d_esc, axis=1)
y_test = np.concatenate(l_esc, axis=0).astype(int)
farr = np.linspace(0,fsog/2,x_test.shape[0])/fsog

for Km in list_K:
    # Random K sampling
    list_finval = []
    for i in range(Nruns):
        xss,farrss = pc_randK(x_test,farr,Km)
        test_dataset = ESC_pc_ss(x = xss,y = y_test, farr = farrss)
        data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)

        losses, total, correct = [], 0, 0
        for imgs,lbls in data_loader_test:
            if(lbls.shape[0] < batch_size):
                continue
            imgs = imgs.cuda()
            lbls = lbls.cuda()
            preds = model(imgs)
            total += lbls.shape[0]
            correct += (preds.argmax(dim=1) == lbls).sum().item()

        avg_loss, avg_acc = np.mean(losses), correct / total
        list_finval.append(avg_acc)
        print("RandK, ", Km)
        print(f"Test acc {avg_acc:.3f}")
        del total, correct, avg_loss,avg_acc
    list_finval = np.array(list_finval)
    dict_errs_randK["data"][(int)(Km)] = [np.mean(list_finval),np.var(list_finval)]

    # Max K sampling
    list_finval = []
    for i in range(1):
        xss,farrss = pc_maxK(x_test,farr,Km)
        test_dataset = ESC_pc_ss(x = xss,y = y_test, farr = farrss)
        data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)

        losses, total, correct = [], 0, 0
        for imgs,lbls in data_loader_test:
            if(lbls.shape[0] < batch_size):
                continue
            imgs = imgs.cuda()
            lbls = lbls.cuda()
            preds = model(imgs)          
            total += lbls.shape[0]
            correct += (preds.argmax(dim=1) == lbls).sum().item()
        avg_loss, avg_acc = np.mean(losses), correct / total
        list_finval.append(avg_acc)
        print("maxK, ", Km)
        print(f"Test acc {avg_acc:.3f}")
        del total, correct, avg_loss,avg_acc
    list_finval = np.array(list_finval)
    dict_errs_maxK["data"][(int)(Km)] = [list_finval[-1],0]

json_file = './paper_plots/FST_randK_expt2.json'
with open(json_file, 'w') as outfile:
    json.dump(dict_errs_randK, outfile)

json_file = './paper_plots/FST_maxK_expt2.json'
with open(json_file, 'w') as outfile:
    json.dump(dict_errs_maxK, outfile)
