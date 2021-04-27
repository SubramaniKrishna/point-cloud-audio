"""
Utility Functions
"""

# Counting and display number of trainable parameters in the model
from prettytable import PrettyTable
def count_parameters(model):
    """
    Input: Pytorch Model
    """
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

# Point Cloud subsampling functions for FB model experiments
import numpy as np
# Max K subsampling
def pc_maxK(x,farr,Kmax):
    """
    Inputs
    ------
    x: array(N,T), N - Window Size, T - Number of frames
    Spectral Frame Vectors
    farr: array(N//2)
		Frequency coordinates for the Spectral point cloud
    Kmax: int
    Number of points to keep in sub-sampled spectrum

    Outputs
    -------
    subsampled_x, subsampled_x_fs: Use as inputs to the FB subsampled dataloader
    """
    subsampled_x = []
    subsampled_x_fs = []
    for i in range(x.shape[1]):
        indices = (-x[:,i]).argsort()[:Kmax]
        xthresh_s = x[:,i][indices]
        fthresh = farr[indices]
        subsampled_x.append(xthresh_s.reshape(1,xthresh_s.shape[0]))
        subsampled_x_fs.append(fthresh.reshape(1,fthresh.shape[0]))

    subsampled_x = np.concatenate(subsampled_x,axis = 0).T
    subsampled_x_fs = np.concatenate(subsampled_x_fs,axis = 0).T

    return subsampled_x,subsampled_x_fs

# Random K subsampling
def pc_randK(x,farr,Kmax):
    """
    Inputs
    ------
    x: array(N,T), N - Window Size, T - Number of frames
    Spectral Frame Vectors
    farr: array(N//2)
		Frequency coordinates for the Spectral point cloud
    Kmax: int
    Number of points to keep in sub-sampled spectrum

    Outputs
    -------
    subsampled_x, subsampled_x_fs: Use as inputs to the FB subsampled dataloader
    """
    subsampled_x = []
    subsampled_x_fs = []
    for i in range(x.shape[1]):
        indices = np.random.permutation(x[:,i].shape[0])[:Kmax]
        xthresh_s = x[:,i][indices]
        fthresh = farr[indices]
        subsampled_x.append(xthresh_s.reshape(1,xthresh_s.shape[0]))
        subsampled_x_fs.append(fthresh.reshape(1,fthresh.shape[0]))

    subsampled_x = np.concatenate(subsampled_x,axis = 0).T
    subsampled_x_fs = np.concatenate(subsampled_x_fs,axis = 0).T

    return subsampled_x,subsampled_x_fs

# Funtions to replace the non-max/non-random chosen elements (as an equivalent experiment to the point cloud sub-sampling experiment)
# Replacement to ensure size consistentcy with baseline NN
def pc_maxK_replace(x,Kmax):
    xreplace = []
    for i in range(x.shape[1]):
        temp = np.zeros(x[:,i].shape[0])
        indices = (-x[:,i]).argsort()[:Kmax]
        temp[indices] = x[:,i][indices]
        xreplace.append(temp)

    xreplace = np.array(xreplace).T
    return xreplace

def pc_randK_replace(x,Kmax):
    xreplace = []
    for i in range(x.shape[1]):
        temp = np.zeros(x[:,i].shape[0])
        indices = np.random.permutation(x[:,i].shape[0])[:Kmax]
        temp[indices] = x[:,i][indices]
        xreplace.append(temp)

    xreplace = np.array(xreplace).T
    return xreplace