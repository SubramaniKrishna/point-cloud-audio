# Baseline classifier for ESC-10 for comparison
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import csv

import torch
torch.manual_seed(0)
import torch.nn as nn
from torchinfo import summary
from dataset import *
from data_processing import *

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

import wandb
wandb.init(project="audio-point-clouds", entity="krishnasubramani")

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
    def __init__(self, layer_dims,nclasses, p = 0.5):
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

# Load ESC-50 filenames into list
list_audio_locs,l = load_esc()
list_train,l_train,list_test,l_test = tt_split(list_audio_locs,l,f = 0.8)
# print(len(list_audio_locs),len(list_train),len(list_test))

# Compute STFT and load data into frames
batch_size = 128
Nfft = 2048
hf = 0.5
tDb = 60

# With the new Train Test Split(splitting the audio entirely, not just the frames)
# Train
d_esc = []
l_esc = []
for i in tqdm(range(len(list_train))):
    fi = list_train[i]
    x, fs = librosa.load(fi,sr = 44100)
    x, index = librosa.effects.trim(x,top_db = tDb)
    x = librosa.stft(x,n_fft = Nfft, win_length = Nfft, hop_length=(int)(Nfft*hf), window = 'hann')/Nfft
    a = np.log(1.0e-8 + np.abs(x))
    # a = np.random.randn(*a.shape)
    d_esc.append(a)
    l_esc.append(l_train[i]*np.ones(a.shape[1]))
x_train = np.concatenate(d_esc, axis=1)
y_train = np.concatenate(l_esc, axis=0).astype(int)

# Test
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

train_dataset = ESC_baseline(x = x_train,y = y_train)
data_loader_train = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)
test_dataset = ESC_baseline(x = x_test,y = y_test)
data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)

# Old train test split (which would split frames)
"""
d_esc = []
l_esc = []
tDb = 60
for i in tqdm(range(len(list_audio_locs))):
    fi = list_audio_locs[i]
    x, fs = librosa.load(fi,sr = 44100)
    x, index = librosa.effects.trim(x,top_db = tDb)
    x = librosa.stft(x,n_fft = Nfft, win_length = Nfft, hop_length=(int)(Nfft*hf), window = 'hann')/Nfft
    a = np.log(1.0e-8 + np.abs(x))
    # a = np.random.randn(*a.shape)
    d_esc.append(a)
    l_esc.append(l[i]*np.ones(a.shape[1]))

x = np.concatenate(d_esc, axis=1)
# Normalize x
# x = x / abs(x).max(axis=0)
y = np.concatenate(l_esc, axis=0).astype(int)
# idx = torch.tensor(y)
# y = torch.zeros(len(idx), idx.max()+1).scatter_(1, idx.unsqueeze(1), 1.).data.numpy()
# y = torch.nn.functional.one_hot(torch.from_numpy(y).long(),num_classes = 10).data.numpy()

# print(x.shape,y.shape)
# Normalized frequency (f/fs)
farr = np.linspace(0,fs/2,Nfft//2 + 1)

Nd = y.shape[0]
# random_indices = np.random.permutation(Nd)
# Ntrain = (int)(0.8*Nd)
# Ntest = Nd - Ntrain

# x_train = x[:,random_indices[:Ntrain]]
# y_train = y[random_indices[:Ntrain]]

# x_test = x[:,random_indices[Ntrain:]]
# y_test = y[random_indices[Ntrain:]]

dataset = ESC_baseline(x = x,y = y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
data_loader_train = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)
data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)
"""

# Training NN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 1000
learning_rate = 1.0e-3
layers = [Nfft//2 + 1,Nfft//4 + 1, Nfft//8]
dropout_prob = 0.5

baseline_net = baseline_ff(layer_dims = layers, nclasses = 10, p = dropout_prob).to(device)
nparams = count_parameters(baseline_net)
criterion = nn.CrossEntropyLoss()
wd = 1.0e-3
optimizer = torch.optim.Adam(baseline_net.parameters(), lr=learning_rate, weight_decay = wd)
wandb.watch(baseline_net)

# msummary = summary(baseline_net, input_size=(batch_size,Nfft//2 + 1))
# print(msummary)
loss_f_epoch = []
for epoch in range(num_epochs):
    baseline_net.train()
    losses, total, correct = [], 0, 0
    for iteration, (yi,xi) in enumerate(data_loader_train):
        yi, xi = yi.to(device, non_blocking=True), xi.to(device, non_blocking=True)
        # print(xi.shape,yi.shape)
        pi = baseline_net(xi.float())
        # print(pi.shape,yi.shape)
        loss = criterion(pi, yi.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        total += yi.shape[0]
        correct += (pi.argmax(dim=1) == yi).sum().item()
    avg_loss, avg_acc = np.mean(losses), correct / total
    loss_f_epoch.append(avg_loss)
    print(f"Epoch {epoch}: train loss {avg_loss:.3f} train acc {avg_acc:.3f}")
    wandb.log({"Training Accuracy": avg_acc, "Training Loss": avg_loss})

    if epoch % 10 == 0:
        baseline_net.eval()
        losses, total, correct = [], 0, 0
        for iteration, (yi,xi) in enumerate(data_loader_test):
            yi, xi = yi.to(device, non_blocking=True), xi.to(device, non_blocking=True)
            # print(xi.shape,yi.shape)
            pi = baseline_net(xi.float())
            # print(pi.shape,yi.shape)
            loss = criterion(pi, yi.long())
            losses.append(loss.item())
            total += yi.shape[0]
            correct += (pi.argmax(dim=1) == yi).sum().item()
        avg_loss, avg_acc = np.mean(losses), correct / total
        print(f"Epoch {epoch}: TEST LOSS {avg_loss:.3f} TEST ACC {avg_acc:.3f}")
        wandb.log({"Test Accuracy": avg_acc, "Test Loss": avg_loss})
# print(x.shape,y.shape)

# pyp.figure()
# pyp.plot(loss_f_epoch)
# pyp.show()
# pyp.savefig('./trainloss.png')
# pyp.close()

# Wandb logging
config = dict(
epochs=num_epochs,
weight_decay = wd,
window_size = Nfft,
hop_factor = hf,
trim_dB = tDb,
sampling_rate = fs,
classes=10,
layers=layers,
batch_size=batch_size,
learning_rate=learning_rate,
dataset="ESC10",
architecture="Feed Forward",
dropout_prob = dropout_prob,
model_params = nparams)
wandb.config.update(config)

# Saving the model weights
dir_pth_save = './baselineparams/'
try: 
    os.makedirs(dir_pth_save, exist_ok = True) 
    print("Directory '%s' created successfully" %dir_pth_save) 
except OSError as error: 
    print("Directory '%s' exists") 

now = datetime.now()
dir_network = dir_pth_save + 'baseline(' + str(now) + ')_net.pth'
torch.save(baseline_net.state_dict(), dir_network)
torch.save(baseline_net.state_dict(), os.path.join(wandb.run.dir, 'baseline(' + str(now) + ')_net.pth'))
wandb.save(os.path.join(wandb.run.dir, 'baseline(' + str(now) + ')_net.pth'))

# descriptor_file = dir_pth_save + 'descriptor_pthfiles.csv'

# params_write = [now, Nfft, hf, fs, layers, batch_size, num_epochs, learning_rate]

# with open(descriptor_file, 'a') as csvFile:
    # writer = csv.writer(csvFile)
    # writer.writerow(params_write)
# csvFile.close()

# I've added this line from CNN, checking to see if sync works!!
# I've added this line from the iMac, checking to see if sync works!!
# HMM Sync Check counter
# Sync works!!
# Windows Sync Check!!

