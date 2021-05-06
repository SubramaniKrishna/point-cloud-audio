"""
Code to generate plots (put in paper)
"""

import json
import matplotlib.pyplot as pyp
import numpy as np
import matplotlib.gridspec as gridspec
import itertools
import math

"""
Experiment 1: Accuracy variation with varying N,Fs
"""
# Framewise Model
data_baseline = 'FB_expt1.json'
data_settransformer = 'FST_expt1.json'
with open(data_baseline) as json_file:
    dict_baseline = json.load(json_file)

with open(data_settransformer) as json_file:
    dict_settransformer = json.load(json_file)

list_N_settransformer = dict_settransformer["list_N"]
list_N_baseline = dict_baseline["list_N"]
fig = pyp.figure(constrained_layout=True)
gs = fig.add_gridspec(2, 2)
fig.add_subplot(gs[0:, 0])
pyp.grid(True)
for F in dict_baseline["data"].keys():
    pyp.plot(list_N_baseline[:-2],dict_baseline["data"][F][:-2],'.-')
    pyp.text(list_N_baseline[4], dict_baseline["data"][F][4], str(int(float(F))), fontsize=8,
        verticalalignment='bottom')
pyp.ylim([0.1,0.7])
pyp.xlim([1000,4200])
pyp.tick_params(axis='y', which='both', labelleft=False, labelright=True)
pyp.ylabel('Accuracy', labelpad=-220)
pyp.xlabel("Window Size (Samples)")
pyp.axvspan(2048, 4200, facecolor='gray', alpha=0.5)
pyp.text(2300, 0.42, " Baseline cannot\n process inputs\n larger than\n training window\n size", fontsize=7.5,
        verticalalignment='top')
pyp.gca().yaxis.tick_right()
pyp.title("FB")
fig.add_subplot(gs[0:, 1])
pyp.grid(True)
for F in dict_settransformer["data"].keys():
    pyp.plot(list_N_settransformer[:-2],dict_settransformer["data"][F][:-2],'.-')
    pyp.text(list_N_settransformer[1], dict_settransformer["data"][F][1], str(int(float(F))), fontsize=8.5,
        verticalalignment='bottom')
pyp.ylim([0.1,0.7])
pyp.xlim([1000,4200])
pyp.title("FST")
pyp.xlabel("Window Size (Samples)")

pyp.savefig("./framewise_N_Fs_varying.pdf", transparent = 'True', bbox_inches = 'tight')

# Temporal Model
data_baseline = 'CNNTemp_expt1.json'
data_settransformer = '3ST_expt1.json'

with open(data_baseline) as json_file:
    dict_baseline = json.load(json_file)

with open(data_settransformer) as json_file:
    dict_settransformer = json.load(json_file)

list_N_settransformer = dict_settransformer["list_N"]
list_N_baseline = dict_baseline["list_N"]
fig = pyp.figure(constrained_layout=True)
gs = fig.add_gridspec(2, 2)
fig.add_subplot(gs[0:, 0])
pyp.grid(True)
for F in dict_baseline["data"].keys():
    pyp.plot(list_N_baseline[:-2],dict_baseline["data"][F][:-2],'.-')
    pyp.text(list_N_baseline[4], dict_baseline["data"][F][4], str(int(float(F))), fontsize=8,
        verticalalignment='bottom')
pyp.ylim([0.1,0.7])
pyp.xlim([500,2200])
pyp.tick_params(axis='y', which='both', labelleft=False, labelright=True)
pyp.ylabel('Accuracy', labelpad=-220)
pyp.xlabel("Window Size (Samples)")
pyp.axvspan(1024, 4200, facecolor='gray', alpha=0.5)
pyp.text(1200, 0.42, " Baseline cannot\n process inputs\n larger than\n training window\n size", fontsize=7.5,
        verticalalignment='top')
pyp.gca().yaxis.tick_right()
pyp.title("CNN")
fig.add_subplot(gs[0:, 1])
pyp.grid(True)
for F in dict_settransformer["data"].keys():
    pyp.plot(list_N_settransformer[:-2],dict_settransformer["data"][F][:-2],'.-')
    pyp.text(list_N_settransformer[1], dict_settransformer["data"][F][1], str(int(float(F))), fontsize=8.5,
        verticalalignment='bottom')
pyp.ylim([0.1,0.7])
pyp.xlim([500,2200])
pyp.title("3ST")
pyp.xlabel("Window Size (Samples)")

pyp.savefig("./temporal_N_Fs_varying.pdf", transparent = 'True', bbox_inches = 'tight')

"""
Experiment 2: Accuracy vs Sub-sampling (fraction of input kept)
"""
# Framewise Models
Ntot = 1024
# Set Transformer
topK = 'FST_maxK_expt2.json'
randK = 'FST_randK_expt2.json'

with open(topK) as json_file:
    dict_tK = json.load(json_file)

with open(randK) as json_file:
    dict_rK = json.load(json_file)

list_tK_st =  dict_tK["list_K"]
list_tK_st = np.divide(list_tK_st,Ntot)
list_rK_st =  dict_rK["list_K"]
list_rK_st = np.divide(list_rK_st,Ntot)
topK_acc_st = []
topK_var_st = []
randK_acc_st = []
randK_var_st = []
for i in dict_tK["data"].keys():
    topK_acc_st.append(dict_tK["data"][i][0])
    topK_var_st.append(math.sqrt(dict_tK["data"][i][1]))
for i in dict_rK["data"].keys():
    if(i == "list_K"):
        continue
    randK_acc_st.append(dict_rK["data"][i][0])
    randK_var_st.append(math.sqrt(dict_rK["data"][i][1]))

# Baseline
topK = 'FB_maxK_expt2.json'
randK = 'FB_randK_expt2.json'

with open(topK) as json_file:
    dict_tK = json.load(json_file)
with open(randK) as json_file:
    dict_rK = json.load(json_file)
list_tK_b =  dict_tK["list_K"]
list_tK_b = np.divide(list_tK_b,Ntot)
list_rK_b =  dict_rK["list_K"]
list_rK_b = np.divide(list_rK_b,Ntot)
topK_acc_b = []
topK_var_b = []
randK_acc_b = []
randK_var_b = []
for i in dict_tK["data"].keys():
    topK_acc_b.append(dict_tK["data"][i][0])
    topK_var_b.append(math.sqrt(dict_tK["data"][i][1]))
for i in dict_rK["data"].keys():
    randK_acc_b.append(dict_rK["data"][i][0])
    randK_var_b.append(math.sqrt(dict_rK["data"][i][1]))
pyp.figure()
pyp.xlabel("Fraction of Points Kept")
pyp.ylabel("Accuracy")
pyp.plot(list_tK_st,topK_acc_st,label = "FST Top")
pyp.errorbar(list_rK_st,randK_acc_st,yerr = randK_var_st, label = "FST Rand")
pyp.plot(list_tK_b,topK_acc_b,label = "FB Top")
pyp.errorbar(list_rK_b,randK_acc_b,yerr = randK_var_b, label = "FB Rand")
pyp.legend(loc = 'best')
pyp.grid(True)
pyp.savefig("./subsampling_framewise.pdf", transparent = 'True', bbox_inches = 'tight')

# Temporal Models
Ntot = 512*10
# Set Transformer
topK = '3ST_maxK_expt2.json'
randK = '3ST_randK_expt2.json'

with open(topK) as json_file:
    dict_tK = json.load(json_file)

with open(randK) as json_file:
    dict_rK = json.load(json_file)

list_tK_st =  dict_tK["list_K"]
list_tK_st = np.divide(list_tK_st,Ntot)
list_rK_st =  dict_rK["list_K"]
list_rK_st = np.divide(list_rK_st,Ntot)
topK_acc_st = []
topK_var_st = []
randK_acc_st = []
randK_var_st = []
for i in dict_tK["data"].keys():
    topK_acc_st.append(dict_tK["data"][i][0])
    topK_var_st.append(math.sqrt(dict_tK["data"][i][1]))
for i in dict_rK["data"].keys():
    if(i == "list_K"):
        continue
    randK_acc_st.append(dict_rK["data"][i][0])
    randK_var_st.append(math.sqrt(dict_rK["data"][i][1]))

# Baseline
topK = 'CNNTemp_maxK_expt2.json'
randK = 'CNNTemp_randK_expt2.json'

with open(topK) as json_file:
    dict_tK = json.load(json_file)
with open(randK) as json_file:
    dict_rK = json.load(json_file)
list_tK_b =  dict_tK["list_K"]
list_tK_b = np.divide(list_tK_b,Ntot)
list_rK_b =  dict_rK["list_K"]
list_rK_b = np.divide(list_rK_b,Ntot)
topK_acc_b = []
topK_var_b = []
randK_acc_b = []
randK_var_b = []
for i in dict_tK["data"].keys():
    topK_acc_b.append(dict_tK["data"][i][0])
    topK_var_b.append(math.sqrt(dict_tK["data"][i][1]))
for i in dict_rK["data"].keys():
    randK_acc_b.append(dict_rK["data"][i][0])
    randK_var_b.append(math.sqrt(dict_rK["data"][i][1]))
pyp.figure()
pyp.xlabel("Fraction of Points Kept")
pyp.ylabel("Accuracy")
pyp.plot(list_tK_st,topK_acc_st,label = "3ST Top")
pyp.errorbar(list_rK_st,randK_acc_st,yerr = randK_var_st, label = "3ST Rand")
pyp.plot(list_tK_b,topK_acc_b,label = "CNN Top")
pyp.errorbar(list_rK_b,randK_acc_b,yerr = randK_var_b, label = "CNN Rand")
pyp.legend(loc = 'best')
pyp.grid(True)
pyp.savefig("./subsampling_temporal.pdf", transparent = 'True', bbox_inches = 'tight')