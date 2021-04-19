import numpy as np
np_seed = 1
torch_seed = 1
np.random.seed(np_seed)
import torch
torch.manual_seed(torch_seed)
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import csv

import librosa
from tqdm import trange
from tqdm import tqdm

# import torch
# import torch.nn as nn
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

import sys
sys.path.append('../set_transformer-master/')


# from data_modelnet40 import ModelFetcher
from modules import ISAB, PMA, SAB

import wandb
wandb.init(project="audio-point-clouds", entity="krishnasubramani")

# Load ESC-50 filenames into list
list_audio_locs,l = load_esc()
list_train,l_train,list_test,l_test = tt_split(list_audio_locs,l,f = 0.8)

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

# Normalized frequency (f/fs)
farr = np.linspace(0,fs/2,Nfft//2 + 1)/fs

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

train_dataset = ESC_pc(x = x_train,y = y_train, farr = farr)
data_loader_train = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)
test_dataset = ESC_pc(x = x_test,y = y_test, farr = farr)
data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)

"""
d_esc = []
l_esc = []
for i in tqdm(range(len(list_audio_locs))):
    fi = list_audio_locs[i]
    x, fs = librosa.load(fi,sr = 44100)
    x, index = librosa.effects.trim(x,top_db = tDb)
    x = librosa.stft(x,n_fft = Nfft, win_length = Nfft, hop_length=(int)(Nfft*hf), window = 'hann')/Nfft
    # x = stft(x, s = Nfft/6,hf = hf)
    a = np.log(1.0e-8 + np.abs(x))
    # a = np.random.randn(*a.shape)
    d_esc.append(a)
    # print(a.shape)
    l_esc.append(l[i]*np.ones(a.shape[1]))

x = np.concatenate(d_esc, axis=1)
# Normalize x
# x = x / abs(x).max(axis=0)
y = np.concatenate(l_esc, axis=0).astype(int)
# idx = torch.tensor(y)
# y = torch.zeros(len(idx), idx.max()+1).scatter_(1, idx.unsqueeze(1), 1.).data.numpy()
# y = torch.nn.functional.one_hot(torch.from_numpy(y).long(),num_classes = 10).data.numpy()
dataset = ESC_pc(x = x, y = y, farr = farr)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
data_loader_train = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)
data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)

"""

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

dhidden = 32
nheads = 8
ninds = 32

model = SetTransformer(dim_hidden=dhidden, num_heads=nheads, num_inds=ninds).to(device)
nparams = count_parameters(model)

# msummary = summary(model, input_size=(batch_size,Nfft//2 + 1,2))
# print(msummary)

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1.0e-3)
criterion = nn.CrossEntropyLoss()
wd = 1.0e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = wd)

# criterion = F.cross_entropy
model = nn.DataParallel(model)
# model = model.cuda()
wandb.watch(model)

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
sampling_rate = fs,
classes=10,
dhidden = dhidden,
nheads = nheads,
ninds = ninds,
batch_size=batch_size,
learning_rate=learning_rate,
dataset="ESC10",
architecture="Set Transformer Framewise",
numpy_seed = np_seed,
torch_seed = torch_seed,
model_params = nparams)
wandb.config.update(config)

# Saving the model weights
dir_pth_save = './settransformerparams/'
try: 
    os.makedirs(dir_pth_save, exist_ok = True) 
    print("Directory '%s' created successfully" %dir_pth_save) 
except OSError as error: 
    print("Directory '%s' exists") 

now = datetime.now()
dir_network = dir_pth_save + 'settransformerpc(' + str(now) + ')_net.pth'
torch.save(model.state_dict(), dir_network)
torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'settransformerpc(' + str(now) + ')_net.pth'))
wandb.save(os.path.join(wandb.run.dir, 'settransformerpc(' + str(now) + ')_net.pth'))

# descriptor_file = dir_pth_save + 'descriptor_pthfiles.csv'

# params_write = [now, Nfft, hf, fs, dhidden, nheads, ninds, batch_size, num_epochs, learning_rate]

# with open(descriptor_file, 'a') as csvFile:
    # writer = csv.writer(csvFile)
    # writer.writerow(params_write)
# csvFile.close()
