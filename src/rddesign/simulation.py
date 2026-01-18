import torch, numpy as np

def linear_sim(ndraws = 5000, cutoff = 0, seed = 10042002):
    gen = torch.Generator().manual_seed(seed)
    U = -0.5 + torch.bernoulli(0.5 * torch.ones((ndraws, 1)), generator = gen) + torch.randn((ndraws, 1), generator = gen)/2
    V = - torch.log(torch.rand((ndraws, 1), generator = gen))
    
    D = U + (1/10) * V + (U < 0) * torch.randn((ndraws, 1), generator = gen)/2
    Z = U + (1/10) * V +  torch.randn((ndraws, 1), generator = gen)/2
    
    W = 1 + U + torch.randn((ndraws, 1), generator = gen)/2
    Y = (D >= 0) * -0.5 + (D < 0) * 0.5 + D + U + torch.randn((ndraws, 1), generator = gen)/2
    return Y, W, D, Z, U
    
