"""
Pytorch Data Loaders for FB, FST, CNN_temp, 3ST
"""

import numpy as np
import torch
from torch.utils.data import Dataset

# FB Dataloader
class ESC_baseline(Dataset):
	"""
	Inputs
	------
		x: array(N,T), N - Window Size, T - Number of frames
		Spectral Frame Vectors
		y: integer(T)
		Label corresponding to each frame
	"""
	def __init__(self, x,y):
		self.x = x
		self.labels = y

	def __len__(self):
		return self.x.shape[1]

	def __getitem__(self, idx):
		return self.labels[idx], torch.tensor(self.x[:,idx])

# FST Dataloader
class ESC_pc(Dataset):
	"""
	Inputs
	------
		x: array(N,T), N - Window Size, T - Number of frames
		Spectral Frame Vectors
		y: integer(T)
		Label corresponding to each frame
		farr: array(N//2)
		Frequency coordinates for the Spectral point cloud
	"""
	def __init__(self, x,y,farr):
		self.x = x
		self.labels = y
		self.farr = farr

	# Bogus value
	def __len__(self):
		return self.x.shape[1]

	def __getitem__(self,idx):
		pc = np.concatenate((self.farr.reshape(1,self.farr.shape[0]),self.x[:,idx].reshape(1,self.x.shape[0])),axis = 0).T
		lbl = self.labels[idx]

		return torch.from_numpy(pc).float(), torch.tensor(lbl)

# FST Dataloader for the subsampling experiments
# Complements the pc_maxK and pc_randK functions (i.e. takes as input x,farr as outputs of these)
class ESC_pc_ss(Dataset):
	"""
	Inputs
	------
		x,farr: Output of pc_maxK, pc_randK
		y: integer(T)
		Label corresponding to each frame
	"""
	def __init__(self, x,y,farr):
		self.x = x
		self.labels = y
		self.farr = farr

	# Bogus value
	def __len__(self):
		return self.x.shape[1]

	def __getitem__(self,idx):
		pc = np.concatenate((self.farr[:,idx].reshape(1,self.farr.shape[0]),self.x[:,idx].reshape(1,self.x.shape[0])),axis = 0).T
		lbl = self.labels[idx]

		return torch.from_numpy(pc).float(), torch.tensor(lbl)

# CNN_temp dataloader
class ESC_baseline_temporal(Dataset):
	"""
	Inputs
	------
		x: array(N,Nt,T), N - Window Size, Nt - Number of frames, T - Number of samples
		The input spectrograms
		y: integer(T)
		Label corresponding to each frame
	"""
	def __init__(self, x,y):
		self.x = x
		self.labels = y

	def __len__(self):
		return self.x.shape[2]

	def __getitem__(self, idx):
		return torch.tensor(self.labels[idx]), torch.tensor(self.x[:,:,idx]).T

# CNN_temp dataloader for subsampling experiments
class ESC_baseline_temporal_maxK(Dataset):
	"""
	Inputs
	------
		x: array(N,Nt,T), N - Window Size, Nt - Number of frames, T - Number of samples
		The input spectrograms
		y: integer(T)
		Label corresponding to each frame
		K: int
		Number of points to keep in sub-sampled spectrum
		flag: "max" or "rand"
		If "max", will keep max K points, if "rand", will keep random K points
	"""
	def __init__(self, x,y,K,flag = "max"):
		self.x = x
		self.labels = y
		self.K = K
		self.flag = flag

	def __len__(self):
		return self.x.shape[2]

	def __getitem__(self, idx):
		xt = self.x[:,:,idx]
		xreplace = np.zeros_like(xt)
		tinds = np.repeat(np.arange(xt.shape[1]),xt.shape[0])
		finds = np.tile(np.arange(xt.shape[0]), xt.shape[1])
		pc = np.vstack((np.vstack((finds,tinds)),xt[finds,tinds])).T
		if(self.flag == "rand"):
			indices = np.random.permutation(pc.shape[0])[:self.K]
		else:
			indices = (-pc[:,-1]).argsort()[:self.K]
		xreplace[finds[indices],tinds[indices]] = xt[finds[indices],tinds[indices]]
		return torch.tensor(self.labels[idx]), torch.tensor(xreplace).T

# 3ST dataloader
class ESC_pc_temp(Dataset):
	"""
	Inputs
	------
		x: array(N,Nt,T), N - Window Size, Nt - Number of frames, T - Number of samples
		The input spectrograms
		y: integer(T)
		Label corresponding to each frame
		farr: array(N//2)
		Frequency coordinates for the Spectrotemporal point cloud
		tarr: array(Nt)
		Temporal coordinates for the Spectrotemporal point cloud
	"""
	def __init__(self, x,y,farr,tarr):
		self.x = x
		self.labels = y
		self.farr = farr
		self.tarr = tarr

	def __len__(self):
		return self.labels.shape[0]

	def __getitem__(self,idx):
		xt = self.x[:,:,idx]
		tinds = np.repeat(np.arange(self.tarr.shape[0]),self.farr.shape[0])
		finds = np.tile(np.arange(self.farr.shape[0]), self.tarr.shape[0])
		pc = np.vstack((np.vstack((self.farr[finds],self.tarr[tinds])),xt[finds,tinds])).T
		lbl = self.labels[idx]
		return torch.from_numpy(pc).float(), torch.tensor(lbl)

# 3ST dataloader for max K subsampling
class ESC_pc_temp_maxKSS(Dataset):
	"""
	Inputs
	------
		x: array(N,Nt,T), N - Window Size, Nt - Number of frames, T - Number of samples
		The input spectrograms
		y: integer(T)
		Label corresponding to each frame
		farr: array(N//2)
		Frequency coordinates for the Spectrotemporal point cloud
		tarr: array(Nt)
		Temporal coordinates for the Spectrotemporal point cloud
		K: int
		Number of points to keep (max K points will be kept)
	"""
	def __init__(self, x,y,farr,tarr,K):
		self.x = x
		self.labels = y
		self.farr = farr
		self.tarr = tarr
		self.K = K

	def __len__(self):
		return self.labels.shape[0]

	def __getitem__(self,idx):
		xt = self.x[:,:,idx]
		tinds = np.repeat(np.arange(self.tarr.shape[0]),self.farr.shape[0])
		finds = np.tile(np.arange(self.farr.shape[0]), self.tarr.shape[0])
		pc = np.vstack((np.vstack((self.farr[finds],self.tarr[tinds])),xt[finds,tinds])).T
		indices = (-pc[:,-1]).argsort()[:self.K]
		pc = pc[indices,:]

		return torch.tensor(pc), torch.tensor(self.labels[idx])

# 3ST dataloader for random K subsampling
class ESC_pc_temp_randKSS(Dataset):
	"""
	Inputs
	------
		x: array(N,Nt,T), N - Window Size, Nt - Number of frames, T - Number of samples
		The input spectrograms
		y: integer(T)
		Label corresponding to each frame
		farr: array(N//2)
		Frequency coordinates for the Spectrotemporal point cloud
		tarr: array(Nt)
		Temporal coordinates for the Spectrotemporal point cloud
		K: int
		Number of points to keep (Random K points will be kept)
	"""
	def __init__(self, x,y,farr,tarr,K):
		self.x = x
		self.labels = y
		self.farr = farr
		self.tarr = tarr
		self.K = K

	def __len__(self):
		return self.labels.shape[0]

	def __getitem__(self,idx):
		xt = self.x[:,:,idx]
		tinds = np.repeat(np.arange(self.tarr.shape[0]),self.farr.shape[0])
		finds = np.tile(np.arange(self.farr.shape[0]), self.tarr.shape[0])
		pc = np.vstack((np.vstack((self.farr[finds],self.tarr[tinds])),xt[finds,tinds])).T
		indices = np.random.permutation(pc.shape[0])[:self.K]
		pc = pc[indices,:]
		
		return torch.tensor(pc), torch.tensor(self.labels[idx])