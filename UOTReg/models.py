import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import random

import torch.distributions as distrib
import torch.distributions.transforms as transform
from torch.distributions import constraints
    
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, n_outputs, bn=False, dropout_rate=.1):
        super().__init__()
        assert 0 <= dropout_rate < 1
        self.input_size = input_size
        h_sizes = [self.input_size] + [hidden_size for _ in range(n_hidden)] + [n_outputs]

        self.hidden = nn.ModuleList()
        self.bn = bn

        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        if bn:
            self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(n_hidden)])

    def forward(self, x):
        if self.bn:
            for i, layer in enumerate(self.hidden[:-1]):
                x = layer(x)
                if i < len(self.bn_layers)-1:  # Apply batch normalization to hidden layers only
                    x = self.bn_layers[i](x)
                x = self.relu(x)
                x = self.dropout(x)
        else:
            for layer in self.hidden[:-1]:
                x = layer(x)
                x = self.relu(x)
                x = self.dropout(x)
        a = self.hidden[-1](x)
        return a

# class FFNN(nn.Module):
#     def __init__(self, input_size, hidden_size, n_hidden, n_outputs, bn=False, dropout_rate=0.1):
#         """
#         Feed-Forward Neural Network.
#         Args:
#             input_size (int): Number of input features.
#             hidden_size (int): Number of units in each hidden layer.
#             n_hidden (int): Number of hidden layers.
#             n_outputs (int): Number of output units.
#             bn (bool): Whether to use batch normalization.
#             dropout_rate (float): Dropout rate (0 <= dropout_rate < 1).
#         """
#         super().__init__()
#         assert 0 <= dropout_rate < 1
#         self.input_size = input_size
#         self.bn = bn

#         # Define layer sizes
#         h_sizes = [self.input_size] + [hidden_size for _ in range(n_hidden)] + [n_outputs]

#         # Create hidden layers
#         self.hidden = nn.ModuleList([nn.Linear(h_sizes[i], h_sizes[i + 1]) for i in range(len(h_sizes) - 1)])

#         # Create batch normalization layers (if enabled)
#         if bn:
#             self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(n_hidden)])

#         # Activation and dropout
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=dropout_rate)

#     def forward(self, x):
#         for i, layer in enumerate(self.hidden[:-1]):
#             x = layer(x)
#             if self.bn and i < len(self.bn_layers):  # Apply batch normalization if enabled
#                 x = self.bn_layers[i](x)
#             x = self.relu(x)
#             if i < len(self.hidden) - 2:  # Skip Dropout for the second-to-last layer
#                 x = self.dropout(x)

#         # Final layer without activation or dropout
#         a = self.hidden[-1](x)
#         return a

class TaskIndependentLayers(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, n_outputs, output_size, bn = False, dropout_rate=.1):
        super().__init__()
        self.n_outputs = n_outputs
        self.task_nets = nn.ModuleList()
        for _ in range(n_outputs):
            self.task_nets.append(FFNN(input_size, hidden_size, n_hidden, output_size, bn, dropout_rate))

    def forward(self, x):
        a = torch.cat(tuple(task_model(x) for task_model in self.task_nets), dim=1)
        return a


class HardSharing_T(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, hidden_out, n_outputs, output_size, n_task_specific_layers=0, task_specific_hidden_size=None, bn = False, dropout_rate=.1):
        super().__init__()
        if task_specific_hidden_size is None:
            task_specific_hidden_size = hidden_size

        self.model = nn.Sequential()
        self.model.add_module('hard_sharing', FFNN(input_size, hidden_size, n_hidden, hidden_out, bn, dropout_rate))

        if n_task_specific_layers > 0:
            self.model.add_module('relu', nn.ReLU())
            self.model.add_module('dropout', nn.Dropout(p=dropout_rate))

        self.model.add_module('task_specific', TaskIndependentLayers(hidden_out, task_specific_hidden_size, n_task_specific_layers, 
                                                                     n_outputs, output_size, bn, dropout_rate))

    def forward(self, x):
        return self.model(x)

class HardSharing_D(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, n_outputs, bn = False, dropout_rate=.1):
        super().__init__()
        # input size = DIM & n_outputs = k

        self.n_outputs = n_outputs
        self.task_nets = nn.ModuleList()
        for _ in range(n_outputs):
            self.task_nets.append(FFNN(input_size, hidden_size, n_hidden, 1, bn, dropout_rate))

    def forward(self, x):
        x = torch.chunk(x,self.n_outputs,dim=1)
        a = torch.cat(tuple(task_model(x[i]) for i,task_model in enumerate(self.task_nets)), dim=1)
        return a

class Seperate_D(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, n_outputs, bn = False, dropout_rate=.1):
        super().__init__()
        # input size = DIM & n_outputs = k
        
        # self.total_size = input * n_outputs # 50 * 100 = 5000
        self.n_outputs = n_outputs
        self.task_nets = nn.ModuleList()
        for _ in range(n_outputs):
            self.task_nets.append(FFNN(input_size, hidden_size, n_hidden, 1, bn, dropout_rate))

    def forward(self, x):
        x = torch.chunk(x,self.n_outputs,dim=1)
        a = torch.cat(tuple(task_model(x[i]) for i,task_model in enumerate(self.task_nets)), dim=1)
        return a

class Seperate_T(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, n_outputs, output_size, bn = False, dropout_rate=.1):
        super().__init__()
        # input size = DIM & n_outputs = k

        self.n_outputs = n_outputs
        self.task_nets = nn.ModuleList()
        for _ in range(n_outputs):
            self.task_nets.append(FFNN(input_size, hidden_size, n_hidden, output_size, bn, dropout_rate))

    def forward(self, x):
        a = torch.cat(tuple(task_model(x) for i,task_model in enumerate(self.task_nets)), dim=1)
        return a

class Gnet(nn.Module):
    def __init__(self, input_dim, output_dim, size=256, num_layers=4, dropout_rate=0.01):
        """
        Generator network.
        Args:
            input_dim (int): Dimension of input.
            DIM (int): Dimension of output.
            size (int): Size of hidden layers.
            num_layers (int): Number of hidden layers.
            dropout_rate (float): Dropout rate.
        """
        super(Gnet, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, max(size, 2 * output_dim)))
        layers.append(nn.ReLU(True))
        layers.append(nn.Dropout(dropout_rate))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(max(size, 2 * output_dim), max(size, 2 * output_dim)))
            layers.append(nn.ReLU(True))
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(max(size, 2 * output_dim), max(size, 2 * output_dim)))
        layers.append(nn.ReLU(True))
        layers.append(nn.Linear(max(size, 2 * output_dim), output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Gnet_old(nn.Module):
    def __init__(self, input_dim, DIM, size=256, dropout_rate=0.005):
        super(Gnet_old, self).__init__()
        
        self.network = nn.Sequential(
                nn.Linear(input_dim, max(size, 2 * DIM)),
                nn.ReLU(True),
                nn.Dropout(dropout_rate),
                nn.Linear(max(size, 2 * DIM), max(size, 2 * DIM)),
                nn.ReLU(True),
                nn.Dropout(dropout_rate),
                nn.Linear(max(size, 2 * DIM), max(size, 2 * DIM)),
                nn.ReLU(True),
                nn.Dropout(dropout_rate),
                nn.Linear(max(size, 2 * DIM), max(size, 2 * DIM)),
                nn.ReLU(True),
                nn.Linear(max(size, 2 * DIM), DIM)
            )

    def forward(self, x):
        return self.network(x)


class Discriminator(nn.Module):
    def __init__(self, DIM, size=256, num_layers=4, dropout_rate=0.01):
        """
        Discriminator network.
        Args:
            DIM (int): Dimension of input.
            size (int): Size of hidden layers.
            num_layers (int): Number of hidden layers.
            dropout_rate (float): Dropout rate.
        """
        super(Discriminator, self).__init__()
        
        layers = []
        layers.append(nn.Linear(DIM, size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        for _ in range(num_layers - 2):  # Minus 2 to account for input and final layers
            layers.append(nn.Linear(size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(size, 64))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(64, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class Flow(transform.Transform, nn.Module):
    domain = constraints.real
    codomain = constraints.real

    def __init__(self):
        transform.Transform.__init__(self)
        nn.Module.__init__(self)

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)

    def __hash__(self):
        return nn.Module.__hash__(self)


class PlanarFlow(Flow):
    def __init__(self, dim):
        super().__init__()  # Call the constructors of both Transform and nn.Module
        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.init_parameters()

    def _call(self, z):
        # Forward transformation
        f_z = F.linear(z, self.weight, self.bias)
        return z + self.scale * torch.tanh(f_z)

    def log_abs_det_jacobian(self, z, y=None):
        # Log determinant of the Jacobian
        f_z = F.linear(z, self.weight, self.bias)
        psi = (1 - torch.tanh(f_z) ** 2) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        return torch.log(det_grad.abs() + 1e-9)

# Main class for normalizing flow
class NormalizingFlow(nn.Module):

    def __init__(self, dim, blocks, flow_length, density):
        super().__init__()
        biject = []
        for f in range(flow_length):
            for b_flow in blocks:
                biject.append(b_flow(dim))
        self.transforms = transform.ComposeTransform(biject)
        self.bijectors = nn.ModuleList(biject)
        self.base_density = density
        self.final_density = distrib.TransformedDistribution(density, self.transforms)
        self.log_det = []

    def forward(self, z):
        self.log_det = []
        # Applies series of flows
        for b in range(len(self.bijectors)):
            self.log_det.append(self.bijectors[b].log_abs_det_jacobian(z))
            z = self.bijectors[b](z)
        return z, self.log_det

class VAE(nn.Module):
    
    def __init__(self, encoder, decoder, encoder_dims, latent_dims):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dims = latent_dims
        self.encoder_dims = encoder_dims
        self.mu = nn.Linear(encoder_dims, latent_dims)
        self.sigma = nn.Sequential(
            nn.Linear(encoder_dims, latent_dims),
            nn.Softplus(),
            nn.Hardtanh(min_val=1e-4, max_val=5.))
        self.apply(self.init_parameters)
    
    def init_parameters(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        
    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # Encode the inputs
        z_params = self.encode(x)
        # Obtain latent samples and latent loss
        z_tilde, kl_div = self.latent(x, z_params)
        # Decode the samples
        x_tilde = self.decode(z_tilde)
        return x_tilde, kl_div
    
    def latent(self, x, z_params):
        n_batch = x.size(0)
        # Retrieve mean and var
        mu, sigma = z_params
        # Re-parametrize
        q = distrib.Normal(torch.zeros(mu.shape[1]), torch.ones(sigma.shape[1]))
        z = (sigma * q.sample((n_batch, ))) + mu
        # Compute KL divergence
        kl_div = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        kl_div = kl_div / n_batch
        return z, kl_div

class VAENormalizingFlow(VAE):
    
    def __init__(self, encoder, decoder, flow, encoder_dims, latent_dims):
        super(VAENormalizingFlow, self).__init__(encoder, decoder, encoder_dims, latent_dims)
        self.flow = flow

    def latent(self, x, z_params):
        n_batch = x.size(0)
        # Retrieve set of parameters
        mu, sigma = z_params
        # Re-parametrize a Normal distribution
        q = distrib.Normal(torch.zeros(mu.shape[1]), torch.ones(sigma.shape[1]))
        # Obtain our first set of latent points
        z_0 = (sigma * q.sample((n_batch, ))) + mu
        # Complexify posterior with flows
        z_k, list_ladj = self.flow(z_0)
        # ln p(z_k) 
        log_p_zk = -0.5 * z_k * z_k
        # ln q(z_0)
        log_q_z0 = -0.5 * (sigma.log() + (z_0 - mu) * (z_0 - mu) * sigma.reciprocal())
        #  ln q(z_0) - ln p(z_k)
        logs = (log_q_z0 - log_p_zk).sum()
        # Add log determinants
        ladj = torch.cat(list_ladj, dim=1)
        # ln q(z_0) - ln p(z_k) - sum[log det]
        logs -= torch.sum(ladj)
        return z_k, (logs / float(n_batch))

class Encoder(nn.Module):
    def __init__(self, nin, n_latent, size=256, num_layers=4, dropout_rate=0.05):
        """
        Encoder network.
        Args:
            nin (int): Input dimension.
            n_latent (int): Dimension of the latent space (output of encoder).
            size (int): Size of hidden layers.
            num_layers (int): Number of hidden layers.
            dropout_rate (float): Dropout rate.
        """
        super(Encoder, self).__init__()
        
        layers = []
        layers.append(nn.Linear(nin, size))
        layers.append(nn.ReLU(True))
        layers.append(nn.Dropout(dropout_rate))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(size, size))
            layers.append(nn.ReLU(True))
            layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(size, n_latent))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
