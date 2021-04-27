"""
Pytorch Models: FB, FST, 3ST, CNN_temp
"""

import torch
import torch.nn as nn
import sys
# Point to Set Transformer Repository
sys.path.append('../set_transformer-master/')
from modules import ISAB, PMA, SAB

# Set Transformer (Vary dim_input for using with both the 2D,3D Point Cloud)
class ST(nn.Module):
    """
    Parameters:
    dim_input: Dimensionality of Input Point Cloud 
    num_outputs: num_seeds in PMA
    dim_output: Number of classes
    num_inds: Number of Inducing Poings in ISAB block
    dim_hidden: Dimensionality of intermediate point clouds
    num_heads: Number of heads to compute multihead attention
    """
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
        super(ST, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        return self.dec(self.enc(X)).squeeze()

# Framewize feedforward baseline
class baseline_ff(nn.Module):
    """
    Parameters:
    layer_dims: List containing the dimensions of consecutive layers(Including input layer{dim(X)} and excluding final layer)
    nclasses: Number of classes
    p: Dropout probability
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

# Temporal Baseline CNN
class CNN_classifier(nn.Module):
    """
    Simple CNN Classifier on the input spectrogram: [batch_size,seq_length,num_features]

    Parameters:
    ----------
        layer_dims: list[integers]
            Layers of classifier are l[0],l[1],l[2],...,l[-1]
        nclass: integer
            Final number of classes, equal to the dimensionality of final layer of classifier
    """

    def __init__(self, Nt,Nf,layer_dims,nclass,p = 0.5):
        super(CNN_classifier, self).__init__()

        self.cnn = nn.Conv2d(1, 1, (Nt, Nf + 1 - layer_dims[0]), stride=(1, 1), padding=(0, 0))
        # Dropout to emulate rand sampling
        self.dpout = nn.Dropout(p=p)

        self.linear = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            self.linear.add_module(name = "Encoder_Layer_{:d}".format(i), module = nn.Linear(in_size, out_size))
            self.linear.add_module(name = "Activation_{:d}".format(i), module = nn.LeakyReLU())
        self.linear.add_module(name = "Logits", module = nn.Linear(layer_dims[-1], nclass))

    def forward(self, x):
        cnn_outp = self.cnn(self.dpout(x.unsqueeze(1)))
        logits = self.linear(cnn_outp.squeeze())
        return logits