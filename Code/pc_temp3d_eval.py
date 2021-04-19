import matplotlib.pyplot as pyp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import csv
from scipy.signal import butter,sosfilt, decimate
import librosa
from tqdm import trange
from tqdm import tqdm


import torch
import torch.nn as nn
from torchinfo import summary
from dataset import *
from data_processing import *

import glob
import sys
sys.path.append('../set_transformer-master/')
from modules import ISAB, PMA, SAB
import yaml
import json
from datetime import datetime
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
now = datetime.now()

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

run_id = 'run-20210415_211413-8ujp3d4j'
# Point to directory of wandb parameters (config.yaml)
pth_yml = glob.glob('./wandb/' + run_id + '/files/*.yaml')[0]
pth_pth = glob.glob('./wandb/' + run_id + '/files/*.pth')[0]

with open(pth_yml) as f:
    dict_params = yaml.safe_load(f)

np_seed = dict_params['np_seed']['value']
torch_seed = dict_params['torch_seed']['value']
np.random.seed(np_seed)
torch.manual_seed(torch_seed)

# Load ESC-50 filenames into list
list_audio_locs,l = load_esc()
nclass = max(l) + 1
list_train,l_train,list_test,l_test = tt_split(list_audio_locs,l,f = 0.8)

Nfft = dict_params['window_size']['value']
hf = dict_params['hop_factor']['value']
Ntemp = dict_params['Ntemp']['value']
tDb = dict_params['trim_dB']['value']

dhidden = dict_params['dhidden']['value']
nheads = dict_params['nheads']['value']
ninds = dict_params['ninds']['value']

model = SetTransformer(dim_input = 3,dim_hidden=dhidden, num_heads=nheads, num_inds=ninds,dim_output = nclass).to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(pth_pth,map_location = device))

count_parameters(model)

# Varying N,Fs

# list_Fs = [44100]
criterion = nn.CrossEntropyLoss()
fsog = dict_params['sampling_rate']['value']
list_Fs = [fsog,0.95*fsog,0.9*fsog,0.8*fsog,0.5*fsog,0.1*fsog]
list_N = [2*Nfft,(int)(1.5*Nfft),(int)(1.25*Nfft),(int)(1.05*Nfft),Nfft,(int)(0.95*Nfft),(int)(0.9*Nfft),(int)(0.8*Nfft),(int)(0.7*Nfft),(int)(0.6*Nfft),(int)(0.5*Nfft),(int)(0.25*Nfft),(int)(0.1*Nfft)]
# list_Fs = [fsog,0.95*fsog]
# list_N = [2*Nfft,(int)(1.5*Nfft)]
list_NF = []
list_finval = []
dict_errs = {"data":{k:[] for k in list_Fs}}
dict_errs["list_Fs"] = list_Fs
dict_errs["list_N"] = list_N
for F in list_Fs:
    for N in list_N:
        Nfft = N
        fs = F
        print(F,N)
        d_esc = []
        l_esc = []
        for i in tqdm(range(len(list_test))):
            fi = list_test[i]
            x, fsog = librosa.load(fi,sr = fsog)
            x, index = librosa.effects.trim(x,top_db = tDb)
            x = librosa.resample(x, fsog, fs, res_type='kaiser_fast',scale= True)
            # x = librosa.stft(x,n_fft = Nfft, win_length = Nfft, hop_length=(int)(Nfft*hf), window = 'hann')/Nfft
            x = librosa.stft(x,n_fft = (int)(2**(np.ceil(np.log2(Nfft)))),win_length = Nfft, hop_length=(int)(Nfft*hf), window = 'hann')/Nfft
            # x = librosa.feature.melspectrogram(sr=fs, S=np.abs(x), n_fft=Nfft, hop_length=(int)(Nfft*hf), win_length=Nfft, n_mels=nmels, window='hann', center=True, pad_mode='reflect', power=2.0)
            # Deleting last fft entry because of positional encoding glitch (wants even #FFT as input)
            x = x[:-1,:] 
            a = np.log(1.0e-8 + np.abs(x))
            # a = x
            b = np.hsplit(a,np.arange(0,x.shape[1],Ntemp))
            
            for ss in b:
                if(ss.shape[1] < Ntemp):
                    continue
                # ss = np.concatenate((init_e.reshape(nmels,1),ss),axis = 1)
                d_esc.append(ss)
                l_esc.append(l_test[i])
            # a = np.random.randn(*a.shape)
            # d_esc.append(a)
            # l_esc.append(l[i])
        # Nfft = nmels
        # x = np.concatenate(d_esc, axis=1)
        x = np.dstack(d_esc)
        # print(x.shape,len(l_esc))
        y = np.array(l_esc).astype(int)
        farr = np.linspace(0,fs/2,x.shape[0])/fs
        tarr = np.linspace(0,((hf*Nfft)/fs)*Ntemp,Ntemp)

        batch_size = 2
        test_dataset = ESC_pc_temp(x = x, y = y, farr = farr,tarr = tarr)
        data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)


        losses, total, correct = [], 0, 0
        for imgs,lbls in data_loader_test:
            # print(imgs.shape,lbls.shape)
            if(lbls.shape[0] < 2):
                continue
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
        # list_NF.append([N,F])
        # list_finval.append([avg_loss,avg_acc])
        dict_errs["data"][F].append(avg_acc)
        # print(F,N)
        print(f"Test acc {avg_acc:.3f}")
        del total, correct, avg_loss,avg_acc
json_file = './plots/pc_3dtemp_' + run_id + '_data.json'
with open(json_file, 'w') as outfile:
    json.dump(dict_errs, outfile)


# PC Subsampling

Nfft = dict_params['window_size']['value']
fsog = dict_params['sampling_rate']['value']
Ntemp = dict_params['Ntemp']['value']
list_K = np.arange(10,(Nfft//2*Ntemp),100)
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
    x = x[:-1,:] 
    a = np.log(1.0e-8 + np.abs(x))
    # a = np.abs(x)
    b = np.hsplit(a,np.arange(0,x.shape[1],Ntemp))
    
    for ss in b:
        if(ss.shape[1] < Ntemp):
            continue
        # ss = np.concatenate((init_e.reshape(Nfft//2,1),ss),axis = 1)
        d_esc.append(ss)
        l_esc.append(l_test[i])

x_test = np.dstack(d_esc)
y_test = np.array(l_esc).astype(int)
farr = np.linspace(0,fsog/2,x_test.shape[0])/fsog
tarr = np.linspace(0,((hf*Nfft)/fsog)*Ntemp,Ntemp)
batch_size = 2

for Km in list_K:
    # Random K sampling
    list_finval = []
    for i in range(Nruns):
        # subsampling
        test_dataset = ESC_pc_temp_randKSS(x = x_test,y = y_test, farr = farr,tarr = tarr,K = Km)
        data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)
        
        criterion = nn.CrossEntropyLoss()
        losses, total, correct = [], 0, 0
        for imgs,lbls in data_loader_test:
            if(lbls.shape[0] < 2):
                continue
            imgs = imgs.float().cuda()
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
    for i in range(Nruns):
        # subsampling
        test_dataset = ESC_pc_temp_maxKSS(x = x_test,y = y_test, farr = farr,tarr = tarr,K = Km)
        data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)

        criterion = nn.CrossEntropyLoss()
        losses, total, correct = [], 0, 0
        for imgs,lbls in data_loader_test:
            if(lbls.shape[0] < 2):
                continue
            imgs = imgs.float().cuda()
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
    dict_errs_maxK["data"][(int)(Km)] = [np.mean(list_finval),np.var(list_finval)]

json_file = './plots/pc_3dtemp_randKSS_' + run_id + '_data.json'
with open(json_file, 'w') as outfile:
    json.dump(dict_errs_randK, outfile)

json_file = './plots/pc_3dtemp_maxKSS_' + run_id + '_data.json'
with open(json_file, 'w') as outfile:
    json.dump(dict_errs_maxK, outfile)
