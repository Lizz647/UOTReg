import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from tqdm import tqdm
from tqdm import tqdm_notebook

import gc
from copy import deepcopy
import matplotlib.pyplot as plt

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

def random_sample(tensor, sample_size):
    # Ensure sample_size does not exceed the first dimension of the tensor
    assert sample_size <= tensor.size(0), "Sample size cannot be larger than the number of samples in the tensor" 
    # Randomly select indices
    indices = torch.randperm(tensor.size(0))[:sample_size]
    # Select and return the random samples
    sampled_tensor = tensor[indices]
    
    return sampled_tensor


def AdvInitialize(G, D1, G_opt, D1_opt, Z_sampler, criterion, datasets_all, 
                  BATCH_SIZE, iterations=20000, lambda_ms=0.1, early_stop_threshold=1e-3, ms=True):
    """
    Perform adversarial initialization for the generator G with optional mode-seeking loss.
    
    Parameters:
        G: Generator network
        D1: Discriminator network
        G_opt: Optimizer for generator
        D1_opt: Optimizer for discriminator
        Z_sampler: Latent vector sampler
        criterion: Loss function
        datasets_all: Real dataset for sampling
        BATCH_SIZE: Batch size
        iterations: Number of training iterations
        lambda_ms: Regularization weight for mode-seeking loss
        early_stop_threshold: Early stopping threshold for generator loss
        ms: Boolean flag to enable/disable mode-seeking loss
    """
    for iteration in range(iterations):
        # ---- Step 1: Train the Discriminator ----
        D1_opt.zero_grad()
        
        # Sample real and fake data
        real_data = random_sample(datasets_all, BATCH_SIZE)
        Z = Z_sampler.sample(BATCH_SIZE).detach()
        fake_data = G(Z)
        
        # Define labels
        real_labels = torch.ones(BATCH_SIZE, 1)
        fake_labels = torch.zeros(BATCH_SIZE, 1)
        
        # Discriminator loss
        D_real_loss = criterion(D1(real_data), real_labels)
        D_fake_loss = criterion(D1(fake_data), fake_labels)
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        D1_opt.step()
        
        # ---- Step 2: Train the Generator ----
        G_opt.zero_grad()
        
        # Generate fake data
        Z = Z_sampler.sample(BATCH_SIZE).detach()
        fake_data = G(Z)
        
        # Generator loss
        G_loss = criterion(D1(fake_data), real_labels)
        
        # Optional: Mode-seeking regularization
        if ms:
            Z1 = Z_sampler.sample(BATCH_SIZE).detach()
            Z2 = Z_sampler.sample(BATCH_SIZE).detach()
            fake_data1 = G(Z1)
            fake_data2 = G(Z2)
            
            output_diff = torch.mean(torch.abs(fake_data1 - fake_data2), dim=1)
            input_diff = torch.mean(torch.abs(Z1 - Z2), dim=1)
            ms_loss = -lambda_ms * torch.mean(output_diff / (input_diff + 1e-8))
            
            # Total generator loss
            total_G_loss = G_loss + ms_loss
        else:
            # Without mode-seeking loss
            total_G_loss = G_loss
        
        total_G_loss.backward()
        G_opt.step()
        
        # ---- Print progress and check early stopping ----
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, D_loss: {D_loss.item()}, G_loss: {G_loss.item()}")
        
        # Uncomment if you want early stopping based on G_loss
        # if G_loss.item() < early_stop_threshold or (ms and total_G_loss.item() < early_stop_threshold):
        #     print(f"Early stopping at iteration {iteration}")
        #     break

    # Final loss
    print(f"Final G_loss: {G_loss.item()}")

def reconstruction_loss_mse(x_tilde, x, average=True):
    loss = F.mse_loss(x_tilde, x, reduction='mean' if average else 'sum')
    return loss

def train_vae(model, optimizer, scheduler, train_loader, model_name='basic', coef=10, epochs=10, plot_it=1):
    # Loss curves
    losses = torch.zeros(epochs, 2)
    # Beta-warmup
    beta = 0

    # Main optimization loop
    for it in range(epochs):
        # Update beta (for beta-VAE warmup)
        beta = 1.0 * (it / float(coef*epochs))
        n_batch = 0.0

        for batch_idx, x in enumerate(train_loader):
            # x is a single tensor of shape (batch_size, 20)
            x = x[0]  # Extract the tensor from the batch tuple if needed
            
            # Pass through VAE
            x_tilde, loss_latent = model(x)

            # Compute reconstruction loss
            loss_recons = reconstruction_loss_mse(x_tilde, x)

            # Evaluate total loss
            loss = loss_recons + (beta * loss_latent)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Accumulate losses
            losses[it, 0] += loss_recons.item()
            losses[it, 1] += loss_latent.item()
            n_batch += 1.0

        # Average the losses over the number of batches
        losses[it, :] /= n_batch

        # Optionally log losses or progress
        if it % plot_it == 0:
            print(f"Epoch {it}/{epochs} - Recon Loss: {losses[it, 0]:.4f}, Latent Loss: {losses[it, 1]:.4f}")

    print("Training Complete.")
    return losses


def optimize_T_and_D(it, G, T, D, T_opt, D_opt, Z_sampler, Ysamplers, NUM, DIM, BATCH_SIZE, ALPHAS_LF, D_ITERS, T_ITERS):
    """
    Optimize T and D iteratively within D_ITERS iterations.
    """
    for d_iter in tqdm(range(D_ITERS)):
        print(f"Joint training: total_iters: {it}, d_iter: {d_iter}")
        it += 1

        # T optimization
        unfreeze(T); freeze(D)
        for t_iter in range(T_ITERS):
            with torch.no_grad():
                X = G(Z_sampler.sample(BATCH_SIZE))
            T_opt.zero_grad()
            T_X = T(X)
            longX = X.repeat(1, NUM)
            T_loss = torch.nn.functional.mse_loss(longX, T_X).mean() - D(T_X).mean()
            T_loss.backward()
            T_opt.step()
        del T_loss, T_X, X
        gc.collect()
        torch.cuda.empty_cache()

        # D optimization
        with torch.no_grad():
            X = G(Z_sampler.sample(BATCH_SIZE))
        Ys = [Ysampler.sample(BATCH_SIZE) for Ysampler in Ysamplers]
        Y = torch.cat(Ys, dim=1)
        
        unfreeze(D); freeze(T)
        T_X = T(X).detach()
        D_opt.zero_grad()
        D_loss = D(T_X).mean() - D(Y).mean()
        D_loss.backward()
        D_opt.step()
        del D_loss, Y, X, T_X
        gc.collect()
        torch.cuda.empty_cache()
    
    return it

def optimize_G(G, T, G_opt, Z_sampler, ALPHAS_LF, NUM, DIM, G_ITERS, BATCH_SIZE_G, G_loss_history):
    """
    Optimize G iteratively.
    """
    G_old = deepcopy(G)
    freeze(G_old)
    unfreeze(G)
    loss_at_start = None

    for g_iter in range(G_ITERS):
        Z = Z_sampler.sample(BATCH_SIZE_G)
        with torch.no_grad():
            G_old_Z = G_old(Z)
            T_G_old_Z = torch.zeros_like(G_old(Z))
        G_old_Z.requires_grad_(True)

        T_X = T(G_old_Z)
        for k in range(NUM):
            T_G_old_Z += ALPHAS_LF[k] * T_X[:, DIM * k : DIM * (k + 1)]
        
        G_opt.zero_grad()
        G_loss = 0.5 * torch.nn.functional.mse_loss(G(Z), T_G_old_Z)
        G_loss.backward()
        G_opt.step()

        if g_iter == 0:
            loss_at_start = G_loss.item()

        G_loss_history.append(G_loss.item())

    del G_old, G_loss, T_G_old_Z, Z
    gc.collect()
    torch.cuda.empty_cache()

    return loss_at_start

def optimize_T_and_D_UOT(it, G, T, D, T_opt, D_opt, Z_sampler, Ysamplers, NUM, DIM, BATCH_SIZE, ALPHAS_LF, D_ITERS, T_ITERS, tau):
    """
    Optimize T and D iteratively within D_ITERS iterations.
    """
    for d_iter in tqdm(range(D_ITERS)):
        print(f"Joint training: total_iters: {it}, d_iter: {d_iter}")
        it += 1

        # T optimization
        unfreeze(T); freeze(D)
        for t_iter in range(T_ITERS):
            with torch.no_grad():
                X = G(Z_sampler.sample(BATCH_SIZE))
            T_opt.zero_grad()
            T_X = T(X)
            longX = X.repeat(1, NUM)
            T_loss = torch.nn.functional.mse_loss(longX, T_X).mean() - D(T_X).mean()
            T_loss.backward()
            T_opt.step()
        del T_loss, T_X, X
        gc.collect()
        torch.cuda.empty_cache()

        # D optimization
        with torch.no_grad():
            X = G(Z_sampler.sample(BATCH_SIZE))
        Ys = [Ysampler.sample(BATCH_SIZE) for Ysampler in Ysamplers]
        Y = torch.cat(Ys, dim=1)
        
        unfreeze(D); freeze(T)
        D_opt.zero_grad()

        # ------------- set up the unbalanced OT loss -------------
        T_X = T(X).detach()
        longX = X.repeat(1, NUM)
        # compute mse vector by vector
        T_X_reshaped = T_X.view(BATCH_SIZE, NUM, DIM)  # Shape: (B, num, d)
        longX_reshaped = longX.view(BATCH_SIZE, NUM, DIM)  # Shape: (B, num, d)
        # Compute the MSE loss for each vector (along the last dimension)
        mse_per_vector = ((T_X_reshaped - longX_reshaped) ** 2).mean(dim=-1)

        DTX =  D(T_X)
        vec_1 = DTX - mse_per_vector
        vec_new = tau * (torch.exp(vec_1 / tau) - 1)

        D_loss = vec_new.mean() - D(Y).mean()
        D_loss.backward()
        D_opt.step()
        del D_loss, Y, X, T_X
        gc.collect()
        torch.cuda.empty_cache()
    
    return it

def optimize_T_and_D_UOT_new(it, G, T, D, T_opt, D_opt, Z_sampler, Ysamplers, NUM, DIM, BATCH_SIZE, ALPHAS_LF, D_ITERS, T_ITERS, tau):
    """
    Optimize T and D iteratively within D_ITERS iterations.
    """
    for d_iter in tqdm(range(D_ITERS)):
        print(f"Joint training: total_iters: {it}, d_iter: {d_iter}")
        it += 1

        # T optimization
        unfreeze(T); freeze(D)
        for t_iter in range(T_ITERS):
            with torch.no_grad():
                X = G(Z_sampler.sample(BATCH_SIZE))
            T_opt.zero_grad()
            T_X = T(X)
            longX = X.repeat(1, NUM)
            T_loss = torch.nn.functional.mse_loss(longX, T_X).mean() - D(T_X).mean()
            T_loss.backward()
            T_opt.step()
        del T_loss, T_X, X
        gc.collect()
        torch.cuda.empty_cache()

        # D optimization
        with torch.no_grad():
            X = G(Z_sampler.sample(BATCH_SIZE))
        Ys = [Ysampler.sample(BATCH_SIZE) for Ysampler in Ysamplers]
        Y = torch.cat(Ys, dim=1)
        
        unfreeze(D); freeze(T)
        D_opt.zero_grad()

        # ------------- set up the unbalanced OT loss -------------
        T_X = T(X).detach()
        # longX = X.repeat(1, NUM)
        # # compute mse vector by vector
        # T_X_reshaped = T_X.view(BATCH_SIZE, NUM, DIM)  # Shape: (B, num, d)
        # longX_reshaped = longX.view(BATCH_SIZE, NUM, DIM)  # Shape: (B, num, d)
        # # Compute the MSE loss for each vector (along the last dimension)
        # mse_per_vector = ((T_X_reshaped - longX_reshaped) ** 2).mean(dim=-1)

        # DTX =  D(T_X)
        # vec_1 = DTX - mse_per_vector
        # vec_new = tau * (torch.exp(vec_1 / tau) - 1)

        DTX =  D(T_X)
        DY = D(Y)
        vec_new = tau * (torch.exp(-DY / tau) - 1)

        D_loss = DTX.mean() + vec_new.mean()
        D_loss.backward()
        D_opt.step()
        del D_loss, Y, X, T_X
        gc.collect()
        torch.cuda.empty_cache()
    
    return it