import torch, numpy as np

def linear_sim(ndraws = 5000, cutoff = 0, seed = 10042002):
    gen = torch.Generator().manual_seed(seed)
    U = 0.25 + torch.bernoulli(0.5 * torch.ones((ndraws, 1)), generator = gen) + torch.randn((ndraws, 1), generator = gen)
    V = torch.log(torch.rand((ndraws, 1), generator = gen))
    
    D = U/2 + (U >= 0) * (-1)
    Z = U/2 + torch.randn((ndraws, 1), generator = gen)/2
    
    W = U/2 + torch.randn((ndraws, 1), generator = gen)/2
    Y = (D >= 0) * 0.5 + (D < 0) * -0.5 - D/2 + U/2 + torch.randn((ndraws, 1), generator = gen)/4
    return Y.flatten().detach().cpu().numpy(), W.flatten().detach().cpu().numpy(), D.flatten().detach().cpu().numpy(), Z.flatten().detach().cpu().numpy(), U.flatten().detach().cpu().numpy()
    
