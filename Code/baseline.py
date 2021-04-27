"""
Framewise Baseline Feed Forward Model Training
"""

# Fixing the seeds for reproducability
np_seed = 0
torch_seed = 0

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
list_train,l_train,list_test,l_test = tt_split(list_audio_locs,l,f = 0.8)

# Compute STFT and load data into frames
batch_size = 128
Nfft = 2048 # Window Size
hf = 0.5 # Hop Factor
tDb = 60 # Librosa trimming audio threshold
fsog = 44100 # Audio Sampling Rate

# Train
d_esc = []
l_esc = []
for i in tqdm(range(len(list_train))):
    fi = list_train[i]
    x, fs = librosa.load(fi,sr = fsog)
    x, index = librosa.effects.trim(x,top_db = tDb)
    x = librosa.stft(x,n_fft = Nfft, win_length = Nfft, hop_length=(int)(Nfft*hf), window = 'hann')/Nfft
    a = np.log(1.0e-8 + np.abs(x))
    d_esc.append(a)
    l_esc.append(l_train[i]*np.ones(a.shape[1]))
x_train = np.concatenate(d_esc, axis=1)
y_train = np.concatenate(l_esc, axis=0).astype(int)

# Test
d_esc = []
l_esc = []
for i in tqdm(range(len(list_test))):
    fi = list_test[i]
    x, fs = librosa.load(fi,sr = fsog)
    x, index = librosa.effects.trim(x,top_db = tDb)
    x = librosa.stft(x,n_fft = Nfft, win_length = Nfft, hop_length=(int)(Nfft*hf), window = 'hann')/Nfft
    a = np.log(1.0e-8 + np.abs(x))
    d_esc.append(a)
    l_esc.append(l_test[i]*np.ones(a.shape[1]))
x_test = np.concatenate(d_esc, axis=1)
y_test = np.concatenate(l_esc, axis=0).astype(int)

train_dataset = ESC_baseline(x = x_train,y = y_train)
data_loader_train = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)
test_dataset = ESC_baseline(x = x_test,y = y_test)
data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)

# Setting up model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 500
learning_rate = 1.0e-3
layers = [Nfft//2 + 1,Nfft//4 + 1, Nfft//8] # Linear Classifier Layers (except for final number of classes)
dropout_prob = 0.5

baseline_net = baseline_ff(layer_dims = layers, nclasses = 10, p = dropout_prob).to(device)
nparams = count_parameters(baseline_net)
criterion = nn.CrossEntropyLoss()
wd = 1.0e-3
optimizer = torch.optim.Adam(baseline_net.parameters(), lr=learning_rate, weight_decay = wd)

for epoch in range(num_epochs):
    baseline_net.train()
    losses, total, correct = [], 0, 0
    for iteration, (yi,xi) in enumerate(data_loader_train):
        yi, xi = yi.to(device, non_blocking=True), xi.to(device, non_blocking=True)
        pi = baseline_net(xi.float())
        loss = criterion(pi, yi.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        total += yi.shape[0]
        correct += (pi.argmax(dim=1) == yi).sum().item()
    avg_loss, avg_acc = np.mean(losses), correct / total
    print(f"Epoch {epoch}: train loss {avg_loss:.3f} train acc {avg_acc:.3f}")

    if epoch % 10 == 0:
        baseline_net.eval()
        losses, total, correct = [], 0, 0
        for iteration, (yi,xi) in enumerate(data_loader_test):
            yi, xi = yi.to(device, non_blocking=True), xi.to(device, non_blocking=True)
            pi = baseline_net(xi.float())
            loss = criterion(pi, yi.long())
            losses.append(loss.item())
            total += yi.shape[0]
            correct += (pi.argmax(dim=1) == yi).sum().item()
        avg_loss, avg_acc = np.mean(losses), correct / total
        print(f"Epoch {epoch}: TEST LOSS {avg_loss:.3f} TEST ACC {avg_acc:.3f}")

# Store all configs to rebuild model
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
architecture="FB (Framewise Feed Forward Baseline)",
dropout_prob = dropout_prob,
model_params = nparams,
numpy_seed = np_seed,
torch_seed = torch_seed)

# Saving the model weights and parameters
now = datetime.now()
dir_pth_save = './model_saves/'
dir_network = dir_pth_save + 'FB(' + str(now) + ')_net.pth'
dir_dictparams = dir_pth_save + 'FB(' + str(now) + ')_config.json'
# Save Weights
torch.save(baseline_net.state_dict(), dir_network)
# Save Config
with open(dir_dictparams, 'w') as fp:
    json.dump(config, fp)

