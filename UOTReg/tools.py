import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from tqdm import tqdm_notebook
from torch.utils.data import TensorDataset

import random

def ewma(x, span=200):
    return pd.DataFrame({'x': x}).ewm(span=span).mean().values[:, 0]

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()    
    
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def gaussian_kernel(u):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u ** 2)

def kernel(x, xi, h):
    """Kernel function K_h(x - xi) with bandwidth h."""
    return (1 / h) * gaussian_kernel((x - xi) / h)

def calculate_moments(X, x, h, j):
    """Calculate the j-th weighted moment \hat{\mu}_j around x."""
    n = len(X)
    weights = np.array([kernel(X[i], x, h) for i in range(n)])
    moment = np.sum(weights * (X - x) ** j) / n
    return moment

def calculate_sigma_squared(mu_0, mu_1, mu_2):
    """Calculate the variance estimate \hat{\sigma}_0^2."""
    return mu_0 * mu_2 - mu_1 ** 2

def calculate_weights(X, x, h, mu_0, mu_1, mu_2):
    """Calculate empirical weights s_in(x, h) for each X_i."""
    n = len(X)
    sigma_0_squared = calculate_sigma_squared(mu_0, mu_1, mu_2)
    weights = [0]*n
    for i in range(n):
        K = kernel(X[i], x, h)
        weights[i] = (1 / sigma_0_squared) * K * (mu_2 - mu_1 * (X[i] - x))
    ALPHAS = [weight / sum(weights) for weight in weights]
    return ALPHAS

def random_sample(tensor, sample_size):
    # Ensure sample_size does not exceed the first dimension of the tensor
    assert sample_size <= tensor.size(0), "Sample size cannot be larger than the number of samples in the tensor" 
    # Randomly select indices
    indices = torch.randperm(tensor.size(0))[:sample_size]
    # Select and return the random samples
    sampled_tensor = tensor[indices]
    
    return sampled_tensor

class PCADataset(Dataset):
    def __init__(self, X, meta):
        self.X = X
        self.meta = meta
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        row = self.X.iloc[index]
        # Assuming the last column is 'ID', use all others as features
        features = torch.tensor(row.values, dtype=torch.float32)
        id = self.meta['donor_id'].iloc[index].tolist()
        # return features, id
        return features, id
    
class tensorDataset(Dataset):
    def __init__(self, X, meta):
        self.X = X
        self.meta = meta
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        row = self.X.iloc[index]
        features = torch.tensor(row.values, dtype=torch.float32)
        return features

class Sampler:
    def __init__(self, device='cuda'):
        self.device = device
    
    def sample(self, size=5):
        pass

class tensorSampler(Sampler):
    def __init__(self, dataset, device='cuda'):
        super().__init__(device=device)
        self.dataset = dataset

    def length(self):
        return len(self.dataset)
        
    def sample(self, batch_size=16):
        ind = random.choices(range(len(self.dataset)), k=batch_size)
        with torch.no_grad():
            batch = self.dataset[ind].clone().to(self.device).float()
        return batch