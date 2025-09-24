import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import gc
from copy import deepcopy
import matplotlib.pyplot as plt

import tqdm
from tqdm import tqdm, tqdm_notebook

import torch
import gc

def UOT_relax_on_2(
    it,
    T,
    D,
    T_opt,
    D_opt,
    Ysampler1,
    Ysampler2,
    BATCH_SIZE,
    D_ITERS,
    T_ITERS,
    tau
):
    """
    Unbalanced OT where we relax the mass constraint on distribution 2.
    Using a KL-like penalty on Y side.
    """
    def freeze(model):
        for p in model.parameters():
            p.requires_grad = False

    def unfreeze(model):
        for p in model.parameters():
            p.requires_grad = True

    for d_iter in tqdm(range(D_ITERS)):
        print(f"UOT (relax on 2): total_iters: {it}, d_iter: {d_iter}")
        it += 1

        # ------------------ T optimization ------------------
        unfreeze(T)
        freeze(D)
        for t_iter in range(T_ITERS):
            with torch.no_grad():
                X = Ysampler1.sample(BATCH_SIZE)
            T_opt.zero_grad()

            T_X = T(X)
            T_loss = torch.nn.functional.mse_loss(X, T_X).mean() - D(T_X).mean()

            T_loss.backward()
            T_opt.step()
        del T_loss, T_X, X
        gc.collect()
        torch.cuda.empty_cache()

        # ------------------ D optimization ------------------
        with torch.no_grad():
            X = Ysampler1.sample(BATCH_SIZE)
            Y = Ysampler2.sample(BATCH_SIZE)

        unfreeze(D)
        freeze(T)
        D_opt.zero_grad()

        T_X = T(X).detach()

        # ---- "Relax on 2" unbalanced term ----
        D_TX = D(T_X).mean()
        D_Y = D(Y)
        vec_new = tau * (torch.exp(- D_Y / tau) - 1)

        D_loss = D_TX + vec_new.mean()

        D_loss.backward()
        D_opt.step()

        del D_loss, Y, X, T_X
        gc.collect()
        torch.cuda.empty_cache()

    return it

def OT(it, T, D, T_opt, D_opt, Ysampler1, Ysampler2, BATCH_SIZE, D_ITERS, T_ITERS):
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
                # X = G(Z_sampler.sample(BATCH_SIZE))
                X = Ysampler1.sample(BATCH_SIZE)
            T_opt.zero_grad()
            T_X = T(X)
            # longX = X.repeat(1, NUM)
            # T_loss = torch.nn.functional.mse_loss(longX, T_X).mean() - D(T_X).mean()
            T_loss = torch.nn.functional.mse_loss(X, T_X).mean() - D(T_X).mean()
            T_loss.backward()
            T_opt.step()
        del T_loss, T_X, X
        gc.collect()
        torch.cuda.empty_cache()

        # D optimization
        with torch.no_grad():
            # X = G(Z_sampler.sample(BATCH_SIZE))
            X = Ysampler1.sample(BATCH_SIZE)
        # Ys = [Ysampler.sample(BATCH_SIZE) for Ysampler in Ysamplers]
        # Y = torch.cat(Ys, dim=1)
        Y = Ysampler2.sample(BATCH_SIZE)
        
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