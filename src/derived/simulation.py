import torch, numpy as np

model_0 = lambda d: (d < 0) * (-1/2) + (d >= 0) * 1/2 + d/2
model_1 = lambda d: (d < 0) * (9.849676 * d**5 + 27.021495 * d**4 + 23.759126 * d**3 + 7.972278 * d**2 + 1.327984 * d + 0.443452) + \
                    (d >= 0) * (4.835652 * d**5 - 11.239543 * d**4 + 9.155945 * d**3 - 3.163629 * d**2 + 0.828011 * d + 0.550595)
model_2 = lambda d: (d < 0) * ( -0.297854 * d**5 - 0.501866 * d**4 + 1.088881 * d**3 + 3.392594 * d**2 + 3.007400 * d + 3.590380) + \
                    (d >= 0) * (14.546066 * d**5 - 59.811812 * d**4 + 89.657718 * d**3 - 60.475018 * d**2 + 18.724299 * d + 0.375062)
model_3 = lambda d: -0.84031627 * d**7 + 1.15154508 * d**6 + 0.17992519 * d**5 + 1.30075625 * d**4 + 2.06973653 * d**3 - 4.72852451 * d**2 + 1.79087152 * d + 0.37626832


def sim_biased(μx, ndraws = 4000, seed = 10042002):
    gen = torch.Generator().manual_seed(seed)
    U = torch.bernoulli(0.45 * torch.ones((ndraws, 1)), generator = gen)
    V = torch.bernoulli(0.95 * torch.ones((ndraws, 1)), generator = gen)
    
    D =  (U == 1) * ( (V == 1) * torch.log(torch.rand((ndraws, 1), generator = gen))/4 + (V == 0) * (torch.randn((ndraws, 1), generator = gen)/4 - 1/4) ) +\
         (U == 0) * ( (V == 1) * (-torch.log(torch.rand((ndraws, 1), generator = gen)))/4 + (V == 0) * (torch.randn((ndraws, 1), generator = gen)/4 + 1/4) )
    Z = 3*U/5 + torch.randn((ndraws, 1), generator = gen)/5 + D/10
    
    W = 3*U/5 + torch.randn((ndraws, 1), generator = gen)/5
    Y = μx(D) - 3*U/5 + torch.randn((ndraws, 1), generator = gen)/5
    return Y.flatten().detach().cpu().numpy(), W.flatten().detach().cpu().numpy(), D.flatten().detach().cpu().numpy(), Z.flatten().detach().cpu().numpy(), U.flatten().detach().cpu().numpy()

def sim_unbiased(μx, ndraws = 4000, seed = 10042002):
    gen = torch.Generator().manual_seed(seed)
    U = torch.bernoulli(0.45 * torch.ones((ndraws, 1)), generator = gen)
    V = torch.bernoulli(0.95 * torch.ones((ndraws, 1)), generator = gen)
    
    D =  (U == 1) * ( (V == 1) * torch.log(torch.rand((ndraws, 1), generator = gen))/4 + (V == 0) * (torch.randn((ndraws, 1), generator = gen)/4 - 1/4) ) +\
         (U == 0) * ( (V == 1) * (-torch.log(torch.rand((ndraws, 1), generator = gen)))/4 + (V == 0) * (torch.randn((ndraws, 1), generator = gen)/4 + 1/4) )
    Z = 3*U/5 + torch.randn((ndraws, 1), generator = gen)/5 + D/10
    
    W = 3*U/5 + torch.randn((ndraws, 1), generator = gen)/5
    Y = μx(D) + torch.randn((ndraws, 1), generator = gen)/5
    return Y.flatten().detach().cpu().numpy(), W.flatten().detach().cpu().numpy(), D.flatten().detach().cpu().numpy(), Z.flatten().detach().cpu().numpy(), U.flatten().detach().cpu().numpy()
