import numpy as np
import torch
from torch.utils.data import Dataset
import torch_geometric.data
from torch_cluster import knn_graph
# import torch_geometric.data


class ESC_baseline(Dataset):
	def __init__(self, x,y):
		self.x = x
		self.labels = y

	def __len__(self):
		return self.x.shape[1]

	def __getitem__(self, idx):
		return self.labels[idx], torch.tensor(self.x[:,idx])

class ESC_pc(Dataset):
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

class ESC_pc_ss(Dataset):
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

class GCN_pc(torch_geometric.data.Dataset):
	def __init__(self, x,y,farr,k):
		self.x = x
		self.labels = y
		self.farr = farr
		self.k = k

	# Bogus value
	def __len__(self):
		return self.x.shape[1]

	def __getitem__(self,idx):
		pc = np.concatenate((self.farr.reshape(1,self.farr.shape[0]),self.x[:,idx].reshape(1,self.x.shape[0])),axis = 0).T
		lbl = self.labels[idx]
		pc = torch.from_numpy(pc).float()
		d = torch_geometric.data.Data(pos = pc, edge_index = knn_graph(pc, k=self.k))
		return d,torch.tensor(lbl)

class ESC_baseline_temporal(Dataset):
	def __init__(self, x,y):
		self.x = x
		self.labels = y

	def __len__(self):
		return self.x.shape[2]

	def __getitem__(self, idx):
		return torch.tensor(self.labels[idx]), torch.tensor(self.x[:,:,idx]).T
		# return torch.tensor(self.labels[idx]), (self.labels[idx] + torch.randn((self.x[:,:,idx].shape[0],self.x[:,:,idx].shape[1]))).T

class ESC_baseline_temporal_maxK(Dataset):
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
		# return torch.tensor(self.labels[idx]), (self.labels[idx] + torch.randn((self.x[:,:,idx].shape[0],self.x[:,:,idx].shape[1]))).T

class ESC_pc_temp(Dataset):
	def __init__(self, x,y,farr,tarr):
		self.x = x
		self.labels = y
		self.farr = farr
		self.tarr = tarr

	# Bogus value
	def __len__(self):
		return self.labels.shape[0]

	def __getitem__(self,idx):
		# print(self.x.shape,self.tarr.shape,self.farr.shape)
		xt = self.x[:,:,idx]
		tinds = np.repeat(np.arange(self.tarr.shape[0]),self.farr.shape[0])
		finds = np.tile(np.arange(self.farr.shape[0]), self.tarr.shape[0])
		pc = np.vstack((np.vstack((self.farr[finds],self.tarr[tinds])),xt[finds,tinds])).T
		# pc = np.concatenate((self.farr.reshape(1,self.farr.shape[0]),self.x[:,idx].reshape(1,self.x.shape[0])),axis = 0).T
		lbl = self.labels[idx]
		# print(pc.shape,lbl.shape)
		return torch.from_numpy(pc).float(), torch.tensor(lbl)

class ESC_pc_temp_maxKSS(Dataset):
	def __init__(self, x,y,farr,tarr,K):
		self.x = x
		self.labels = y
		self.farr = farr
		self.tarr = tarr
		self.K = K

	# Bogus value
	def __len__(self):
		return self.labels.shape[0]

	def __getitem__(self,idx):
		# finds = []
		# tinds = np.repeat(np.arange(self.tarr.shape[0]),self.K)
		# xt = self.x[:,:,idx]
		# for i in range(xt.shape[1]):
		# 	indices = (-xt[:,i]).argsort()[:self.K]
		# 	# indices = np.random.permutation(self.x[:,i,idx].shape[0])[:self.K]
		# 	finds.append(indices)
		# finds = np.dstack(finds,axis = 0)
		# print(tinds.shape,finds.shape)
		# pc = np.vstack((np.vstack((self.farr[finds],self.tarr[tinds])),xt[finds,tinds])).T

		xt = self.x[:,:,idx]
		tinds = np.repeat(np.arange(self.tarr.shape[0]),self.farr.shape[0])
		finds = np.tile(np.arange(self.farr.shape[0]), self.tarr.shape[0])
		pc = np.vstack((np.vstack((self.farr[finds],self.tarr[tinds])),xt[finds,tinds])).T
		indices = (-pc[:,-1]).argsort()[:self.K]
		pc = pc[indices,:]

		return torch.tensor(pc), torch.tensor(self.labels[idx])

class ESC_pc_temp_randKSS(Dataset):
	def __init__(self, x,y,farr,tarr,K):
		self.x = x
		self.labels = y
		self.farr = farr
		self.tarr = tarr
		self.K = K

	# Bogus value
	def __len__(self):
		return self.labels.shape[0]

	def __getitem__(self,idx):
		# finds = []
		# tinds = np.repeat(np.arange(self.tarr.shape[0]),self.K)
		# xt = self.x[:,:,idx]
		# for i in range(xt.shape[1]):
		# 	indices = (-xt[:,i]).argsort()[:self.K]
		# 	# indices = np.random.permutation(self.x[:,i,idx].shape[0])[:self.K]
		# 	finds.append(indices)
		# finds = np.dstack(finds,axis = 0)
		# print(tinds.shape,finds.shape)
		# pc = np.vstack((np.vstack((self.farr[finds],self.tarr[tinds])),xt[finds,tinds])).T

		xt = self.x[:,:,idx]
		tinds = np.repeat(np.arange(self.tarr.shape[0]),self.farr.shape[0])
		finds = np.tile(np.arange(self.farr.shape[0]), self.tarr.shape[0])
		pc = np.vstack((np.vstack((self.farr[finds],self.tarr[tinds])),xt[finds,tinds])).T
		indices = np.random.permutation(pc.shape[0])[:self.K]
		pc = pc[indices,:]
		
		return torch.tensor(pc), torch.tensor(self.labels[idx])

class ESC_pc_settransformerrnn(Dataset):
	def __init__(self, x, y, farr):
		self.x = x
		self.labels = y
		self.farr = farr

	def __len__(self):
		return self.x.shape[2]

	def __getitem__(self, idx):
		farr = self.farr.reshape(self.farr.shape[0],1)
		fA = np.tile(farr,self.x.shape[1])
		pc = np.stack((self.x[:,:,idx],fA))
		# pc = np.concatenate((self.farr.reshape(1,self.farr.shape[0]),self.x[:,idx].reshape(1,self.x.shape[0])),axis = 0).T
		return torch.tensor(pc), torch.tensor(self.labels[idx])

class ESC_pc_settransformerrnn_KSS(Dataset):
	"""
	Sample top K points in the spectrum and return the corresponding (F,M) point cloud
	"""
	def __init__(self, x, y, farr, K):
		self.x = x
		self.labels = y
		self.farr = farr
		self.K = K

	def __len__(self):
		return self.x.shape[2]

	def __getitem__(self, idx):
		subsampled_x = []
		subsampled_x_fs = []
		for i in range(self.x.shape[1]):
			indices = (-self.x[:,i,idx]).argsort()[:self.K]
			xthresh_s = self.x[:,i,idx][indices]
			fthresh = self.farr[indices]
			subsampled_x.append(xthresh_s.reshape(1,xthresh_s.shape[0]))
			subsampled_x_fs.append(fthresh.reshape(1,fthresh.shape[0]))
		subsampled_x = np.concatenate(subsampled_x,axis = 0).T
		subsampled_x_fs = np.concatenate(subsampled_x_fs,axis = 0).T
		# farr = self.farr.reshape(self.farr.shape[0],1)
		# fA = np.tile(farr,self.x.shape[1])
		pc = np.stack((subsampled_x,subsampled_x_fs))
		# print(pc.shape)
		# pc = np.concatenate((self.farr.reshape(1,self.farr.shape[0]),self.x[:,idx].reshape(1,self.x.shape[0])),axis = 0).T
		return torch.tensor(pc), torch.tensor(self.labels[idx])

class ESC_pc_settransformerrnn_RandKSS(Dataset):
	"""
	Sample K Random points in the spectrum and return the corresponding (F,M) point cloud
	"""
	def __init__(self, x, y, farr, K):
		self.x = x
		self.labels = y
		self.farr = farr
		self.K = K

	def __len__(self):
		return self.x.shape[2]

	def __getitem__(self, idx):
		subsampled_x = []
		subsampled_x_fs = []
		for i in range(self.x.shape[1]):
			# indices = (-self.x[:,i,idx]).argsort()[:self.K]
			indices = np.random.permutation(self.x[:,i,idx].shape[0])[:self.K]
			xthresh_s = self.x[:,i,idx][indices]
			fthresh = self.farr[indices]
			subsampled_x.append(xthresh_s.reshape(1,xthresh_s.shape[0]))
			subsampled_x_fs.append(fthresh.reshape(1,fthresh.shape[0]))
		subsampled_x = np.concatenate(subsampled_x,axis = 0).T
		subsampled_x_fs = np.concatenate(subsampled_x_fs,axis = 0).T
		# farr = self.farr.reshape(self.farr.shape[0],1)
		# fA = np.tile(farr,self.x.shape[1])
		pc = np.stack((subsampled_x,subsampled_x_fs))
		# print(pc.shape)
		# pc = np.concatenate((self.farr.reshape(1,self.farr.shape[0]),self.x[:,idx].reshape(1,self.x.shape[0])),axis = 0).T
		return torch.tensor(pc), torch.tensor(self.labels[idx])
