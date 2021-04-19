import numpy as np
np_seed = 1
torch_seed = 1
np.random.seed(np_seed)
import torch
torch.manual_seed(torch_seed)
import torch.nn as nn
from datetime import datetime
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import csv

import librosa
from tqdm import trange
from tqdm import tqdm

# import torch
# import torch.nn as nn
from torchinfo import summary
from dataset import *
from data_processing import *

import sys
sys.path.append('../set_transformer-master/')


# from data_modelnet40 import ModelFetcher
from modules import ISAB, PMA, SAB

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

# Load ESC-50 filenames into list
list_audio_locs,l = load_esc()
nclass = max(l) + 1
list_train,l_train,list_test,l_test = tt_split(list_audio_locs,l,f = 0.8)

# Compute STFT and load data into frames
Nfft = 1024
hf = 0.5
tDb = 60
fs = 44100
# nmels = 128
# init_e = np.ones(nmels)*0.2
# Number of temporal context frames to keep (c - N//2 : c + N//2)
Ntemp = 10
farr = np.linspace(0,fs/2,Nfft//2)/fs
tarr = np.linspace(0,((hf*Nfft)/fs)*Ntemp,Ntemp)
batch_size = 16

# Loading Train Data
d_esc = []
l_esc = []
for i in tqdm(range(len(list_train))):
    fi = list_train[i]
    x, fs = librosa.load(fi,sr = fs)
    x, index = librosa.effects.trim(x,top_db = tDb)
    x = librosa.stft(x,n_fft = Nfft, win_length = Nfft, hop_length=(int)(Nfft*hf), window = 'hann')/Nfft
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
        l_esc.append(l_train[i])

x_train = np.dstack(d_esc)
y_train = np.array(l_esc).astype(int)

# print(x[:,:,1].shape,y.shape,farr.shape,tarr.shape)
train_dataset = ESC_pc_temp(x = x_train, y = y_train, farr = farr,tarr = tarr)
data_loader_train = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)

# Loading Test Data
d_esc = []
l_esc = []
for i in tqdm(range(len(list_test))):
    fi = list_test[i]
    x, fs = librosa.load(fi,sr = fs)
    x, index = librosa.effects.trim(x,top_db = tDb)
    x = librosa.stft(x,n_fft = Nfft, win_length = Nfft, hop_length=(int)(Nfft*hf), window = 'hann')/Nfft
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

x_test = np.dstack(d_esc)
y_test = np.array(l_esc).astype(int)

# print(x[:,:,1].shape,y.shape,farr.shape,tarr.shape)
test_dataset = ESC_pc_temp(x = x_test, y = y_test, farr = farr,tarr = tarr)
data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)


# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
# data_loader_train = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)
# data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)
# print(train_size,test_size)


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


# Training NN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 500
learning_rate = 1.0e-3

dhidden = 64
nheads = 8
ninds = 64

model = SetTransformer(dim_input = 3,dim_hidden=dhidden, num_heads=nheads, num_inds=ninds,dim_output = nclass).to(device)
nparams = count_parameters(model)

# msummary = summary(model, input_size=(batch_size,Nfft//2 + 1,2))
# print(msummary)

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1.0e-3)
criterion = nn.CrossEntropyLoss()
wd = 1.0e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = wd)

# criterion = F.cross_entropy
model = nn.DataParallel(model)
wandb.watch(model)
now = datetime.now()
# model = model.cuda()
# wandb.watch(model)

for epoch in range(num_epochs):
    model.train()
    losses, total, correct = [], 0, 0
    # for imgs, _, lbls in generator.train_data():
    for imgs,lbls in data_loader_train:
        # print(imgs.shape)
        # imgs = torch.Tensor(imgs).cuda()
        # lbls = torch.Tensor(lbls).long().cuda()
        # imgs = imgs.cuda()
        # lbls = lbls.cuda()
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        preds = model(imgs)
        # print(preds.shape,lbls.shape)
        loss = criterion(preds, lbls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        total += lbls.shape[0]
        correct += (preds.argmax(dim=1) == lbls).sum().item()

    avg_loss, avg_acc = np.mean(losses), correct / total
    # writer.add_scalar("train_loss", avg_loss)
    # writer.add_scalar("train_acc", avg_acc)
    print(f"Epoch {epoch}: train loss {avg_loss:.3f} train acc {avg_acc:.3f}")
    wandb.log({"Training Accuracy": avg_acc, "Training Loss": avg_loss})

    if epoch % 10 == 0:
        with torch.no_grad():
            model.eval()
            losses, total, correct = [], 0, 0
            # for imgs, _, lbls in generator.test_data():
            for imgs,lbls in data_loader_test:
                # print(imgs.shape)
                # imgs = torch.Tensor(imgs).cuda()
                # lbls = torch.Tensor(lbls).long().cuda()
                imgs = imgs.cuda()
                lbls = lbls.cuda()
                # print(imgs.shape,lbls.shape)
                preds = model(imgs)
                preds = model(imgs)
                loss = criterion(preds, lbls)

                losses.append(loss.item())
                total += lbls.shape[0]
                correct += (preds.argmax(dim=1) == lbls).sum().item()
            avg_loss, avg_acc = np.mean(losses), correct / total
            # writer.add_scalar("test_loss", avg_loss)
            # writer.add_scalar("test_acc", avg_acc)
        print(f"Epoch {epoch}: test loss {avg_loss:.3f} test acc {avg_acc:.3f}")
        wandb.log({"Test Accuracy": avg_acc, "Test Loss": avg_loss})

# Wandb logging
config = dict(
epochs=num_epochs,
weight_decay = wd,
window_size = Nfft,
hop_factor = hf,
trim_dB = tDb,
Ntemp = Ntemp,
sampling_rate = fs,
classes=nclass,
dhidden = dhidden,
nheads = nheads,
ninds = ninds,
batch_size=batch_size,
learning_rate=learning_rate,
dataset="ESC10",
architecture="Set Transformer Temporal (T,F,M) PC",
np_seed = np_seed,
torch_seed = torch_seed,
model_params = nparams)
wandb.config.update(config)

torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'settransformerpc_temp(' + str(now) + ')_net.pth'))
wandb.save(os.path.join(wandb.run.dir, 'settransformerpc_temp(' + str(now) + ')_net.pth'))


