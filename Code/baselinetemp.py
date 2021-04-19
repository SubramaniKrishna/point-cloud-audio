# Baseline classifier for ESC-10 for comparison (including temporal context)
np_seed = 0
torch_seed = 0

import numpy as np
np.random.seed(np_seed)
import scipy as sp
import matplotlib.pyplot as pyp
import librosa
from tqdm import trange
from tqdm import tqdm

from datetime import datetime
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import csv

import torch
import math
torch.manual_seed(torch_seed)
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchinfo import summary
from dataset import *
from data_processing import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import wandb
wandb.init(project="audio-point-clouds", entity="krishnasubramani")

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

# Choice of baseline method: 1 = unrolled FF, 2 = LSTM, 3 = xformer, 4 = CNN
choice_baseline = 4

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

    def __init__(self, Nt,Nf,layer_dims,nclass,p = 0.5):
        super(CNN_classifier, self).__init__()

        self.cnn = nn.Conv2d(1, 1, (Nt, Nf + 1 - layer_dims[0]), stride=(1, 1), padding=(0, 0))
        # Dropout to emulate rand sampling
        self.dpout = nn.Dropout(p=p)

        self.linear = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            self.linear.add_module(name = "Encoder_Layer_{:d}".format(i), module = nn.Linear(in_size, out_size))
            self.linear.add_module(name = "Activation_{:d}".format(i), module = nn.LeakyReLU())
        self.linear.add_module(name = "Logits", module = nn.Linear(layer_dims[-1], nclass))

    def forward(self, x):
        cnn_outp = self.cnn(self.dpout(x.unsqueeze(1)))
        logits = self.linear(cnn_outp.squeeze())
        return logits


# Load ESC-50 filenames into list
list_audio_locs,l = load_esc()
nclass = max(l) + 1
list_train,l_train,list_test,l_test = tt_split(list_audio_locs,l,f = 0.8)


# Compute STFT and load data into frames
Nfft = 1024
hf = 0.5
# Number of temporal context frames to keep (c - N//2 : c + N//2)
Ntemp = 10
tDb = 60
# init_e = np.ones(Nfft//2)*0.2

# With the new Train Test Split(splitting the audio entirely, not just the frames)
# Train
d_esc = []
l_esc = []
for i in tqdm(range(len(list_train))):
    fi = list_train[i]
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
        l_esc.append(l_train[i])

x_train = np.dstack(d_esc)
y_train = np.array(l_esc).astype(int)

# Test
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

batch_size = 128
train_dataset = ESC_baseline_temporal(x = x_train,y = y_train)
data_loader_train = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)
test_dataset = ESC_baseline_temporal(x = x_test,y = y_test)
data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)

"""
d_esc = []
l_esc = []
tDb = 60
for i in tqdm(range(len(list_audio_locs))):
    fi = list_audio_locs[i]
    x, fs = librosa.load(fi,sr = 44100)
    x, index = librosa.effects.trim(x,top_db = tDb)
    x = librosa.stft(x,n_fft = Nfft, win_length = Nfft, hop_length=(int)(Nfft*hf), window = 'hann')/Nfft
    # x = librosa.feature.melspectrogram(sr=fs, S=np.abs(x), n_fft=Nfft, hop_length=(int)(Nfft*hf), win_length=Nfft, n_mels=nmels, window='hann', center=True, pad_mode='reflect', power=2.0)
    # Deleting last fft entry because of positional encoding glitch (wants even #FFT as input)
    x = x[:-1,:] 
    a = np.log(1.0e-8 + np.abs(x))
    # a = np.abs(x)
    b = np.hsplit(a,np.arange(0,x.shape[1],Ntemp))
    
    for ss in b:
        if(ss.shape[1] < Ntemp):
            continue
        # ss = np.concatenate((init_e.reshape(Nfft//2,1),ss),axis = 1)
        d_esc.append(ss)
        l_esc.append(l[i])

x = np.dstack(d_esc)
y = np.array(l_esc).astype(int)

dataset = ESC_baseline_temporal(x = x,y = y)
batch_size = 128
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
data_loader_train = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)
data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)
"""


ninp = Nfft//2
if(choice_baseline == 1):
    print("Unrolled FF NN")
    layer_dims = [Ntemp*ninp,Ntemp*ninp//2]
    model = FF_classifier(layer_dims = layer_dims, nclass = nclass).to(device)
    wandb.config.update({
        "architecture" : "Baseline Unrolled Feed Forward",
        "layer_dims" : layer_dims
    })
elif(choice_baseline == 2):
    print("LSTM")
    hidden_dim = 512
    output_dim = 512
    num_layers = 4
    layer_dims = [output_dim,output_dim//2,100]
    dropout = 0
    model = LSTM_classifier(input_dim=ninp, hidden_dim=hidden_dim, batch_size=batch_size, output_dim=output_dim, num_layers = num_layers,layer_dims=layer_dims, nclass = nclass, dropout = dropout).to(device)
    wandb.config.update({
        "architecture" : "Baseline Recurrent (LSTM)",
        "hidden_dim" : hidden_dim,
        "output_dim" : output_dim,
        "num_layers" : num_layers,
        "layer_dims" : layer_dims,
        "dropout": dropout
    })
elif(choice_baseline == 3):
    print("Transformer")
    nhid = 64 
    nlayers = 2 
    nhead = 8 
    layer_dims = [ninp,50]
    max_len = 100
    dropout = 0.2 
    model = baseline_Transformer(ninp, nhead, nhid, nlayers, layer_dims, nclass, max_len,dropout).to(device)
    wandb.config.update({
        "architecture" : "Baseline Transformer (Self Attention)",
        "nhid" : nhid,
        "nlayers" : nlayers,
        "nhead" : nhead,
        "layer_dims" : layer_dims,
        "dropout": dropout,
        "max_len": max_len
    })
else:
    print("CNN")
    Nt = Ntemp
    Nf = ninp
    layer_dims = [ninp,256,100]
    dropout_prob = 0.5
    model = CNN_classifier(Nt, Nf, layer_dims, nclass,dropout_prob).to(device)
    wandb.config.update({
        "architecture" : "Baseline CNN",
        "Nt" : Ntemp,
        "Nf" : Nf,
        "layer_dims" : layer_dims,
        "dropout_prob": dropout_prob
    })

nparams = count_parameters(model)

criterion = nn.CrossEntropyLoss()
wd = 1.0e-3
num_epochs = 500
learning_rate = 1.0e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = wd)
scheduler_step = 50
scheduler_gamma = 0.98
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler_step, gamma=scheduler_gamma)
wandb.watch(model)
now = datetime.now()


for epoch in range(num_epochs):
    model.train()
    losses, total, correct = [], 0, 0
    for iteration, (yi,xi) in enumerate(data_loader_train):
        yi, xi = yi.to(device, non_blocking=True), xi.to(device, non_blocking=True)
        pi = model(xi.float())
        loss = criterion(pi, yi.long())
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        # scheduler.step()
        losses.append(loss.item())
        total += yi.shape[0]
        correct += (pi.argmax(dim=1) == yi).sum().item()
    avg_loss, avg_acc = np.mean(losses), correct / total
    print(f"Epoch {epoch}: train loss {avg_loss:.3f} train acc {avg_acc:.3f}")
    wandb.log({"Training Accuracy": avg_acc, "Training Loss": avg_loss})

    if epoch % 10 == 0:
        model.eval()
        losses, total, correct = [], 0, 0
        for iteration, (yi,xi) in enumerate(data_loader_test):
            yi, xi = yi.to(device, non_blocking=True), xi.to(device, non_blocking=True)
            pi = model(xi.float())
            loss = criterion(pi, yi.long())
            losses.append(loss.item())
            total += yi.shape[0]
            correct += (pi.argmax(dim=1) == yi).sum().item()
        avg_loss, avg_acc = np.mean(losses), correct / total
        print(f"Epoch {epoch}: TEST LOSS {avg_loss:.3f} TEST ACC {avg_acc:.3f}")
        wandb.log({"Test Accuracy": avg_acc, "Test Loss": avg_loss})

wandb.config.update({
    "epochs": num_epochs,
    "weight_decay": wd,
    "window_size": Nfft,
    "hop_factor": hf,
    "trim_dB": tDb,
    "Ntemp": Ntemp,
    "sampling_rate": fs,
    "classes": nclass,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "scheduler_step": scheduler_step,
    "scheduler_gamma": scheduler_gamma,
    "dataset": "ESC10",
    "ninp": Nfft//2,
    "model_params": nparams,
    "np_seed" : np_seed,
    "torch_seed" : torch_seed,
})

torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'baseline_temp(' + str(now) + ')_net.pth'))
wandb.save(os.path.join(wandb.run.dir, 'baseline_temp(' + str(now) + ')_net.pth'))
