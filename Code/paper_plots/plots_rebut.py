"""
Code to generate plots (for the rebuttal)
"""

import json
import matplotlib.pyplot as pyp
import numpy as np
import matplotlib.gridspec as gridspec
import itertools
import math


"""
Rebuttal Experiment: Accuracy vs Sub-sampling done using "importance" sampling
"""
# Temporal Models
Ntot = 512*10
# Set Transformer with naive random/maxK sampling
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

# Temporal Models with importance sampling
Ntot = 512*10
# Set Transformer
topK = '3ST_rebut_expt_maxK.json'
randK = '3ST_rebut_expt_randK.json'

with open(topK) as json_file:
    dict_tK = json.load(json_file)

with open(randK) as json_file:
    dict_rK = json.load(json_file)

list_tK_st_is =  dict_tK["list_K"]
list_tK_st_is = np.divide(list_tK_st_is,Ntot)
list_rK_st_is =  dict_rK["list_K"]
list_rK_st_is = np.divide(list_rK_st_is,Ntot)

list_winF = dict_tK['data'].keys()

pyp.figure()
pyp.xlabel("Fraction of Points Kept")
pyp.ylabel("Accuracy")
pyp.plot(list_tK_st,topK_acc_st,label = "Top")
pyp.errorbar(list_rK_st,randK_acc_st,yerr = randK_var_st, label = "Rand")

for winF in list_winF:
    topK_acc_st_is = []
    topK_var_st_is = []
    randK_acc_st_is = []
    randK_var_st_is = []
    for i in dict_tK["data"][winF].keys():
        topK_acc_st_is.append(dict_tK["data"][winF][i][0])
        topK_var_st_is.append(math.sqrt(dict_tK["data"][winF][i][1]))
    for i in dict_rK["data"][winF].keys():
        if(i == "list_K"):
            continue
        randK_acc_st_is.append(dict_rK["data"][winF][i][0])
        randK_var_st_is.append(math.sqrt(dict_rK["data"][winF][i][1]))
    pyp.plot(list_tK_st_is,topK_acc_st_is,label = "IS Top")
pyp.errorbar(list_rK_st_is,randK_acc_st_is,yerr = randK_var_st_is, label = "IS Rand")
pyp.legend(loc = 'best')
pyp.grid(True)
pyp.savefig("./rebut_expt_is.pdf", transparent = 'True', bbox_inches = 'tight')
