import numpy as np
# np.random.seed(0)
import scipy as sp
import matplotlib.pyplot as pyp
import librosa
from tqdm import trange
from tqdm import tqdm

from datetime import datetime
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import csv
import glob
import yaml

import torch
import math
# torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchinfo import summary
from dataset import *
from data_processing import *
from datetime import datetime
import json
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

# Choice of baseline method: 1 = unrolled FF, 2 = LSTM, 3 = xformer, 4 = CNN
choice_baseline = 4
dict_methods = {1: "Unrolled FF", 2: "LSTM", 3: "Transformer", 4: "CNN"}

# Unrolled FF Classifier
class FF_classifier(nn.Module):
    """
    Simple FF Classifier after unrolling the input (input of size [batch_size,seq_length,num_features])

    Parameters:
    ----------
        layer_dims: list[integers]
            Layers of classifier are l[0],l[1],l[2],...,l[-1]
        nclass: integer
            Final number of classes, equal to the dimensionality of final layer of classifier
    """

    def __init__(self, layer_dims,nclass):
        super(FF_classifier, self).__init__()

        self.ninp = layer_dims[0]

        self.linear = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            self.linear.add_module(name = "Encoder_Layer_{:d}".format(i), module = nn.Linear(in_size, out_size))
            self.linear.add_module(name = "Activation_{:d}".format(i), module = nn.LeakyReLU())
        self.linear.add_module(name = "Logits", module = nn.Linear(layer_dims[-1], nclass))

    def forward(self, x):
        x = x.view(-1, self.ninp)
        logits = self.linear(x)
        return logits

# Recurrent (LSTM) Classifier
class LSTM_classifier(nn.Module):
    """
    LSTM classifier taking as input small spectrograms (input of size [batch_size,seq_length,num_features])

    Parameters:
    ----------
        input_dim: integer
            Dimension of input features
        hidden_dim: integer
            Dimension of LSTM hidden layer
        batch_size: integer
            Batch Size
        output_dim: integer
            Output dimension of LSTM
        num_layers: integer
            Number of LSTM layers applied (sequentially one after the other)
        layer_dims: list[integers]
            Layers of classifierare l[0],l[1],l[2],...,l[-1]
        nclass: integer
            Final number of classes, equal to the dimensionality of final layer of classifier
        dropout: float
            Dropout Probability
    """

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers, layer_dims, nclass, dropout = 0.2):
        super(LSTM_classifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,batch_first = True, dropout = dropout)
        self.linear = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            self.linear.add_module(name = "Encoder_Layer_{:d}".format(i), module = nn.Linear(in_size, out_size))
            self.linear.add_module(name = "Activation_{:d}".format(i), module = nn.LeakyReLU())
        self.linear.add_module(name = "Logits", module = nn.Linear(layer_dims[-1], nclass))

    def init_hidden(self):
        return (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
        )

    def forward(self, input):
        lstm_out, hidden = self.lstm(input)
        logits = self.linear(lstm_out[:,-1,:])
        return logits

# Transformer Classifier
class baseline_Transformer(nn.Module):
    """
    Transformer Classifier using PyTorch Transformer Encoder to capture self attention in input sequence (input of size [batch_size,seq_length,num_features])

    Parameters:
    ----------
        ninp: integer
            Dimension of input features
        nhead: integer
            Number of heads in multi-head attention
        nhid: integer
            Dimension of input in the Transformer encoder feedforward network
        nlayers: integer
            Number of Transformer Encoders applied sequentially
        layer_dims: list[integers]
            Layers of classifierare l[0],l[1],l[2],...,l[-1]
        nclass: integer
            Final number of classes, equal to the dimensionality of final layer of classifier
        max_len: integer
            Max Length of input temporal sequence to consider (should be >= input sequence length)
        dropout: float
            Dropout Probability
        aggr: boolean
            If True, will pool (mean) the output attention values and feed input to classifier
            If False, will consider only the 0th output attention (input should contain the appropriate class tag)
    """
    def __init__(self, ninp, nhead, nhid, nlayers,layer_dims, nclass, max_len = 10, dropout=0.5, aggr = True):
        super(baseline_Transformer, self).__init__()
        self.aggr = aggr
        self.pos_encoder = PositionalEncoding(ninp, dropout, max_len = max_len)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.classifier = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            self.classifier.add_module(name = "Encoder_Layer_{:d}".format(i), module = nn.Linear(in_size, out_size))
            self.classifier.add_module(name = "Activation_{:d}".format(i), module = nn.LeakyReLU())
        self.classifier.add_module(name = "Logits", module = nn.Linear(layer_dims[-1], nclass))
        self.ninp = ninp

    def forward(self, src):
        # Reshape inputs to shape as required by Torch Transformer Encoder
        src = src.permute(1, 0, 2) 
        src = src*math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        if(self.aggr):
            logits = self.classifier(output.mean(axis = 0))
        else:
            logits = self.classifier(output[0,:,:])
        return logits

# Accompanying Position Encoding class for the transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# CNN on the input spectrogram
class CNN_classifier(nn.Module):
    """
    Simple CNN Classifier on the input spectrogram: [batch_size,seq_length,num_features]

    Parameters:
    ----------
        layer_dims: list[integers]
            Layers of classifier are l[0],l[1],l[2],...,l[-1]
        nclass: integer
            Final number of classes, equal to the dimensionality of final layer of classifier
    """

    def __init__(self, Nt,Nf,layer_dims,nclass):
        super(CNN_classifier, self).__init__()

        self.cnn = nn.Conv2d(1, 1, (Nt, Nf + 1 - layer_dims[0]), stride=(1, 1), padding=(0, 0))

        self.linear = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            self.linear.add_module(name = "Encoder_Layer_{:d}".format(i), module = nn.Linear(in_size, out_size))
            self.linear.add_module(name = "Activation_{:d}".format(i), module = nn.LeakyReLU())
        self.linear.add_module(name = "Logits", module = nn.Linear(layer_dims[-1], nclass))

    def forward(self, x):
        cnn_outp = self.cnn(x.unsqueeze(1))
        logits = self.linear(cnn_outp.squeeze())
        return logits

run_id = 'run-20210416_163520-2fgqo2k1'
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

ninp = Nfft//2

if(choice_baseline == 2):
    hidden_dim = dict_params['hidden_dim']['value']
    output_dim = dict_params['output_dim']['value']
    num_layers = dict_params['num_layers']['value']
    layer_dims = dict_params['layer_dims']['value']
    dropout = dict_params['dropout']['value']
    nclass = dict_params['classes']['value']
    batch_size = dict_params['batch_size']['value']
    model = LSTM_classifier(input_dim=ninp, hidden_dim=hidden_dim, batch_size=batch_size, output_dim=output_dim, num_layers = num_layers,layer_dims=layer_dims, nclass = nclass, dropout = dropout).to(device)
if(choice_baseline == 3):
    nhid = dict_params['nhid']['value'] 
    nlayers = dict_params['nlayers']['value'] 
    nhead = dict_params['nhead']['value']
    layer_dims = dict_params['layer_dims']['value']
    max_len = dict_params['max_len']['value']
    dropout = dict_params['dropout']['value'] 
    model = baseline_Transformer(ninp, nhead, nhid, nlayers, layer_dims, nclass, max_len,dropout).to(device)
else:
    Nt = dict_params['Nt']['value'] 
    Nf = dict_params['Nf']['value'] 
    layer_dims = dict_params['layer_dims']['value'] 
    model = CNN_classifier(Nt, Nf, layer_dims, nclass).to(device)


model.load_state_dict(torch.load(pth_pth,map_location = device))
criterion = nn.CrossEntropyLoss()
count_parameters(model)

# Max N,Fs

fsog = dict_params['sampling_rate']['value']
list_Fs = [fsog,0.95*fsog,0.9*fsog,0.8*fsog,0.5*fsog,0.1*fsog]
list_N = [Nfft,(int)(0.95*Nfft),(int)(0.9*Nfft),(int)(0.8*Nfft),(int)(0.7*Nfft),(int)(0.6*Nfft),(int)(0.5*Nfft),(int)(0.25*Nfft),(int)(0.1*Nfft)]
# list_Fs = [fsog,0.95*fsog]
# list_N = [Nfft,(int)(0.9*Nfft)]
list_NF = []
list_finval = []
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
        for i in tqdm(range(len(list_test))):
            fi = list_test[i]
            x, fsog = librosa.load(fi,sr = fsog)
            x, index = librosa.effects.trim(x,top_db = tDb)
            x = librosa.resample(x, fsog, fs, res_type='kaiser_fast',scale= True)
            # x = librosa.stft(x,n_fft = Nfft, win_length = Nfft, hop_length=(int)(Nfft*hf), window = 'hann')/Nfft
            x = librosa.stft(x,n_fft = ninp*2,win_length = Nfft, hop_length=(int)(Nfft*hf), window = 'hann')/(ninp*2)
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
        # farr = np.linspace(0,fs/2,x.shape[0])/fs
        # tarr = np.linspace(0,((hf*Nfft)/fs)*Ntemp,Ntemp)
        batch_size = 2
        test_dataset = ESC_baseline_temporal(x = x,y = y)
        data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)


        losses, total, correct = [], 0, 0
        for lbls,imgs in data_loader_test:
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
json_file = './plots/baseline_temporal_' + dict_methods[choice_baseline] + '_' + run_id + '_data.json'
with open(json_file, 'w') as outfile:
    json.dump(dict_errs, outfile)


# "Modified" sub-sampling

Nfft = dict_params['window_size']['value']
fsog = dict_params['sampling_rate']['value']
Ntemp = dict_params['Ntemp']['value']
list_K = np.arange(10,(Nfft//2*Ntemp),100)
list_K[-1] = (Nfft//2*Ntemp)
# list_K = np.array([(Nfft//2*Ntemp)])
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

batch_size = 2
for Km in list_K:
    # Random K sampling
    list_finval = []
    for i in range(Nruns):
        # subsampling
        test_dataset = ESC_baseline_temporal_maxK(x = x_test,y = y_test, K = Km, flag = "rand")
        data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)
        
        criterion = nn.CrossEntropyLoss()
        losses, total, correct = [], 0, 0
        for lbls,imgs in data_loader_test:
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
        test_dataset = ESC_baseline_temporal_maxK(x = x_test,y = y_test, K = Km, flag = "max")
        data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)

        criterion = nn.CrossEntropyLoss()
        losses, total, correct = [], 0, 0
        for lbls,imgs in data_loader_test:
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

json_file = './plots/baseline_temporal_randK_' + dict_methods[choice_baseline] + '_' + run_id + '_data.json'
with open(json_file, 'w') as outfile:
    json.dump(dict_errs_randK, outfile)

json_file = './plots/baseline_temporal_maxK_' + dict_methods[choice_baseline] + '_' + run_id + '_data.json'
with open(json_file, 'w') as outfile:
    json.dump(dict_errs_maxK, outfile)
