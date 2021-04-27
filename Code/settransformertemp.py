"""
Temporal (3D) Set Transformer Model Training
"""

# Fixing the seeds for reproducability
np_seed = 1
torch_seed = 1

import numpy as np
np.random.seed(np_seed)
import torch
torch.manual_seed(torch_seed)
import torch.nn as nn
from datetime import datetime
# Setting Cuda Devices to Use
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import json
import librosa
from tqdm import trange
from tqdm import tqdm

from dataset import *
from data_processing import *
from utils import count_parameters
from models import *

# Load ESC-50 filenames into list
list_audio_locs,l = load_esc(loc = '../../../../Code/Experiments/GNN/ESC-50-master/meta/esc50.csv',loc_audio = '../../../../Code/Experiments/GNN/ESC-50-master/audio/')
nclass = max(l) + 1
list_train,l_train,list_test,l_test = tt_split(list_audio_locs,l,f = 0.8)

# Compute STFT and load data into frames
Nfft = 1024 # Window Size
hf = 0.5 # Hop Factor
tDb = 60 # Librosa trimming audio threshold
fsog = 44100 # Audio Sampling Rate
Ntemp = 10 # Number of spectral frames to consider (effective temporal context)
farr = np.linspace(0,fsog/2,Nfft//2)/fsog # Frequency coordinates for the STFT point cloud
tarr = np.linspace(0,((hf*Nfft)/fsog)*Ntemp,Ntemp) # Temporal coordinates for the STFT point cloud
batch_size = 16

# Loading Train Data
d_esc = []
l_esc = []
for i in tqdm(range(len(list_train))):
    fi = list_train[i]
    x, fs = librosa.load(fi,sr = fsog)
    x, index = librosa.effects.trim(x,top_db = tDb)
    x = librosa.stft(x,n_fft = Nfft, win_length = Nfft, hop_length=(int)(Nfft*hf), window = 'hann')/Nfft
    x = x[:-1,:] 
    a = np.log(1.0e-8 + np.abs(x))
    b = np.hsplit(a,np.arange(0,x.shape[1],Ntemp))
    
    for ss in b:
        if(ss.shape[1] < Ntemp):
            continue
        d_esc.append(ss)
        l_esc.append(l_train[i])
x_train = np.dstack(d_esc)
y_train = np.array(l_esc).astype(int)
train_dataset = ESC_pc_temp(x = x_train, y = y_train, farr = farr,tarr = tarr)
data_loader_train = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)

# Loading Test Data
d_esc = []
l_esc = []
for i in tqdm(range(len(list_test))):
    fi = list_test[i]
    x, fs = librosa.load(fi,sr = fsog)
    x, index = librosa.effects.trim(x,top_db = tDb)
    x = librosa.stft(x,n_fft = Nfft, win_length = Nfft, hop_length=(int)(Nfft*hf), window = 'hann')/Nfft
    x = x[:-1,:] 
    a = np.log(1.0e-8 + np.abs(x))
    b = np.hsplit(a,np.arange(0,x.shape[1],Ntemp))
    
    for ss in b:
        if(ss.shape[1] < Ntemp):
            continue
        d_esc.append(ss)
        l_esc.append(l_test[i])

x_test = np.dstack(d_esc)
y_test = np.array(l_esc).astype(int)
test_dataset = ESC_pc_temp(x = x_test, y = y_test, farr = farr,tarr = tarr)
data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)

# Setting up model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 500
learning_rate = 1.0e-3

# Model Parameters
dhidden = 64 # Hidden point cloud dimensionality
nheads = 8 # Number of heads for multihead attention
ninds = 64 # Number of inducing points for the ISAB block

model = ST(dim_input = 3,dim_hidden=dhidden, num_heads=nheads, num_inds=ninds,dim_output = nclass).to(device)
nparams = count_parameters(model)

# Setting up NN training
criterion = nn.CrossEntropyLoss()
wd = 1.0e-3 # L2 regularization
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = wd)

# Use with multiple GPUs for faster training
model = nn.DataParallel(model)

for epoch in range(num_epochs):
    model.train()
    losses, total, correct = [], 0, 0
    for imgs,lbls in data_loader_train:
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        preds = model(imgs)
        loss = criterion(preds, lbls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        total += lbls.shape[0]
        correct += (preds.argmax(dim=1) == lbls).sum().item()

    avg_loss, avg_acc = np.mean(losses), correct / total
    print(f"Epoch {epoch}: train loss {avg_loss:.3f} train acc {avg_acc:.3f}")

    if epoch % 10 == 0:
        with torch.no_grad():
            model.eval()
            losses, total, correct = [], 0, 0
            for imgs,lbls in data_loader_test:
                imgs = imgs.cuda()
                lbls = lbls.cuda()
                preds = model(imgs)
                loss = criterion(preds, lbls)

                losses.append(loss.item())
                total += lbls.shape[0]
                correct += (preds.argmax(dim=1) == lbls).sum().item()
            avg_loss, avg_acc = np.mean(losses), correct / total
        print(f"Epoch {epoch}: test loss {avg_loss:.3f} test acc {avg_acc:.3f}")

# Store all configs to rebuild model
config = dict(
epochs=num_epochs,
weight_decay = wd,
window_size = Nfft,
hop_factor = hf,
trim_dB = tDb,
Ntemp = Ntemp,
sampling_rate = fs,
classes = 10,
dhidden = dhidden,
nheads = nheads,
ninds = ninds,
batch_size = batch_size,
learning_rate = learning_rate,
dataset = "ESC10",
architecture = "3ST (Set Transformer Temporal)",
np_seed = np_seed,
torch_seed = torch_seed,
model_params = nparams)

# Saving the model weights and parameters
now = datetime.now()
dir_pth_save = './model_saves/'
dir_network = dir_pth_save + '3ST(' + str(now) + ')_net.pth'
dir_dictparams = dir_pth_save + '3ST(' + str(now) + ')_config.json'
# Save Weights
torch.save(model.state_dict(), dir_network)
# Save Config
with open(dir_dictparams, 'w') as fp:
    json.dump(config, fp)


