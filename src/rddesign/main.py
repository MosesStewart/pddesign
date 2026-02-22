import numpy as np, pandas as pd, warnings, torch, sys, os, re
from scipy.stats import norm
sys.path.append('/'.join(re.split('/|\\\\', os.path.dirname( __file__ ))[0:-1]))
from rddesign.helpers import *
from itertools import permutations
from torchmin import minimize, minimize_constr
from math import factorial, log

class pdd:
    def __init__(self, Y: np.ndarray, W: np.ndarray, D: np.ndarray, Z: np.ndarray, cutoff=0.0, alpha=0.05, kernel='triangle', 
                 bandwidth = None, dtype = torch.float32, device = 'cpu', seed = 10042002):
        self.dtype, self.device = dtype, device
        self.Y = torch.as_tensor(Y, dtype=dtype, device=device)
        if self.Y.ndim == 1: self.Y = self.Y.reshape(-1, 1)
        self.W = torch.as_tensor(W, dtype=dtype, device=device)
        if self.W.ndim == 1: self.W = self.W.reshape(-1, 1)
        self.D = torch.as_tensor(D, dtype=dtype, device=device)
        if self.D.ndim == 1: self.D = self.D.reshape(-1, 1)
        self.Z = torch.as_tensor(Z, dtype=dtype, device=device)
        if self.Z.ndim == 1: self.Z = self.Z.reshape(-1, 1)
        self.n = int(self.D.shape[0])
        self.q = int(self.W.shape[1])
        self.cutoff = torch.tensor(cutoff, dtype=dtype, device=device)
        self.alpha = torch.tensor(alpha, dtype=dtype, device=device)
        self.𝛾 = None
        if kernel == 'triangle': 
            self.kernel = triangular_kernel
            self.ρ = 0.850
        elif kernel == 'rectangle': 
            self.kernel = rectangle_kernel
            self.ρ = 1
        else: 
            self.kernel = epanechnikov_kernel
            self.ρ = 0.898
        if type(bandwidth) != type(None):
            self.custom_bandwidth = torch.as_tensor(bandwidth, dtype=dtype, device=device).flatten()
        else:
            self.custom_bandwidth = None
        self.h = {'+': 3 * torch.std(self.D) * self.n**(-1/4), '-': 3 * torch.std(self.D) * self.n**(-1/4)}
        self.b = {'+': 1/self.ρ * self.h['+'], '-': 1/self.ρ * self.h['-']}
        self.gen = torch.Generator(device = device).manual_seed(seed)
        self.M = 2 * self.n * int(log(self.n))
        self.I, self.J = self.__sample_perms(self.n, self.M, self.device, self.gen)

    def __sample_perms(self, n: int, nsamples: int, device = 'cpu', gen = torch.Generator()) -> torch.Tensor:
        # random set of permutations when computing mean of edgeworth terms
        N = n * (n - 1)
        nsamples = min(nsamples, N)

        k = torch.randperm(N, device = device, dtype = torch.int32, generator = gen)[:nsamples] 
        I = k // (n - 1)
        jp = k % (n - 1)
        J = jp + (jp >= I).to(torch.int32)
        return I, J

    def __build_matrices(self):
        one_n = torch.ones((self.n, 1), dtype=self.dtype, device=self.device)
        Ih, Ib, Dm = {'+': 1 / self.h['+'], '-': 1 / self.h['-']}, {'+': 1 / self.b['+'], '-': 1 / self.b['-']}, self.D - self.cutoff
        self.R_1 = {'+': torch.cat([one_n, Ih['+'] * Dm], dim=1), '-': torch.cat([one_n, Ih['-'] * Dm], dim=1)}
        self.R_2 = {'+': torch.cat([one_n, Ib['+'] * Dm, (Ib['+'] * Dm)**2], dim=1), '-': torch.cat([one_n, Ib['-'] * Dm, (Ib['-'] * Dm)**2], dim=1)}

        self.ind = {'+': (self.D >= self.cutoff), '-': (self.D < self.cutoff)}
        self.𝜔 = {'+': (Ih['+'] * self.ind['+'] * self.kernel(Ih['+'] * Dm)), '-': (Ih['-'] * self.ind['-'] * self.kernel(Ih['-'] * Dm))}
        self.𝛿 = {'+': (Ib['+'] * self.ind['+'] * self.kernel(Ib['+'] * Dm)), '-': (Ib['-'] * self.ind['-'] * self.kernel(Ib['-'] * Dm))}
        
        self.I_n = torch.eye(self.n, dtype=self.dtype, device=self.device)
        self.Γ_1 = {'+': (1 / self.n) * (self.R_1['+'].T * self.𝜔['+'].T) @ self.R_1['+'], '-': (1 / self.n) * (self.R_1['-'].T * self.𝜔['-'].T) @ self.R_1['-']}
        self.Γ_2 = {'+': (1 / self.n) * (self.R_2['+'].T * self.𝛿['+'].T) @ self.R_2['+'], '-': (1 / self.n) * (self.R_2['-'].T * self.𝛿['-'].T) @ self.R_2['-']}
        self.Γ_1_inv = {'+': torch.linalg.pinv(self.Γ_1['+']), '-': torch.linalg.pinv(self.Γ_1['-'])}
        self.Γ_2_inv = {'+': torch.linalg.pinv(self.Γ_2['+']), '-': torch.linalg.pinv(self.Γ_2['-'])}
        self.Λ_1 = {'+': (1 / self.n) * (self.R_1['+'].T * self.𝜔['+'].T) @ (Ih['+'] * Dm)**2, '-': (1 / self.n) * (self.R_1['-'].T * self.𝜔['-'].T) @ (Ih['-'] * Dm)**2}
        self.Λ_2 = {'+': (1 / self.n) * (self.R_2['+'].T * self.𝛿['+'].T) @ (Ib['+'] * Dm)**2, '-': (1 / self.n) * (self.R_2['-'].T * self.𝛿['-'].T) @ (Ib['-'] * Dm)**2}
        self.Λ_1_2 = {'+': (1 / self.n) * (self.R_1['+'].T * self.𝜔['+'].T) @ (Ih['+'] * Dm)**3, '-': (1 / self.n) * (self.R_1['-'].T * self.𝜔['-'].T) @ (Ih['-'] * Dm)**3}
        self.Λ_2_1 = {'+': (1 / self.n) * (self.R_2['+'].T * self.𝛿['+'].T) @ (Ib['+'] * Dm)**2, '-': (1 / self.n) * (self.R_2['-'].T * self.𝛿['-'].T) @ (Ib['-'] * Dm)**2}
        self.e_0 = torch.tensor([[1.0], [0.0]], dtype=self.dtype, device=self.device)
        self.e_2 = torch.tensor([[0.0], [0.0], [1.0]], dtype=self.dtype, device=self.device)
        self.e_3 = torch.tensor([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0]], dtype=self.dtype, device=self.device)
        
        self.R_5 = {'+': torch.cat([torch.ones((self.n, 1), dtype=self.dtype, device=self.device), (Ih['+'] * Dm), (Ih['+'] * Dm)**2, (Ih['+'] * Dm)**3, (Ih['+'] * Dm)**4, (Ih['+'] * Dm)**5], dim=1),
                    '-': torch.cat([torch.ones((self.n, 1), dtype=self.dtype, device=self.device), (Ih['-'] * Dm), (Ih['-'] * Dm)**2, (Ih['-'] * Dm)**3, (Ih['-'] * Dm)**4, (Ih['-'] * Dm)**5], dim=1)}
        self.Γ_5 = {'+': (1 / self.n) * (self.R_5['+'].T * self.𝛿['+'].T) @ self.R_5['+'], '-': (1 / self.n) * (self.R_5['-'].T * self.𝛿['-'].T) @ self.R_5['-']}
        self.Γ_5_inv = {'+': torch.linalg.pinv(self.Γ_5['+']), '-': torch.linalg.pinv(self.Γ_5['-'])}

        get_B_2β = lambda X, sn: self.Γ_2_inv[sn] @ (self.R_2[sn].T * self.𝛿[sn].T) / self.n @ X
        self.B_2β = {'+': torch.concat([get_B_2β(self.Y, '+')] + [get_B_2β(self.W[:, [j]], '+') for j in range(self.q)], dim = 1), 
                  '-': torch.concat([get_B_2β(self.Y, '-')] + [get_B_2β(self.W[:, [j]], '-') for j in range(self.q)], dim = 1)}
        get_H_1β = lambda X, sn: self.Γ_1_inv[sn] @ (self.R_1[sn].T * self.𝜔[sn].T) / self.n @ X
        self.H_1β = {'+': torch.concat([get_H_1β(self.Y, '+')] + [get_H_1β(self.W[:, [j]], '+') for j in range(self.q)], dim = 1), 
                  '-': torch.concat([get_H_1β(self.Y, '-')] + [get_H_1β(self.W[:, [j]], '-') for j in range(self.q)], dim = 1)}
        self.ε = {'+': torch.concat([self.Y - self.R_1['+'] @ self.H_1β['+'][:, [0]]] + [self.W[:, [j]] - self.R_1['+'] @ self.H_1β['+'][:, [j + 1]] for j in range(self.q)], dim=1),
                  '-': torch.concat([self.Y - self.R_1['-'] @ self.H_1β['-'][:, [0]]] + [self.W[:, [j]] - self.R_1['-'] @ self.H_1β['-'][:, [j + 1]] for j in range(self.q)], dim=1)}  # (n, q+1)
        self.σ = {'+': torch.concat([torch.abs(self.Y - self.R_2['+'] @ self.B_2β['+'][:, [0]])] + [torch.abs(self.W[:, [j]] - self.R_2['+'] @ self.B_2β['+'][:, [j + 1]]) for j in range(self.q)], dim=1),
                  '-': torch.concat([torch.abs(self.Y - self.R_2['-'] @ self.B_2β['-'][:, [0]])] + [torch.abs(self.W[:, [j]] - self.R_2['-'] @ self.B_2β['-'][:, [j + 1]]) for j in range(self.q)], dim=1)}  # (n, q+1)
        self.Σ = {'+': torch.stack([torch.diag((self.σ['+'][:, j]**2)) for j in range(self.q + 1)], dim = 0), # (q + 1, n, n)
                  '-': torch.stack([torch.diag((self.σ['-'][:, j]**2)) for j in range(self.q + 1)], dim = 0),}
        self.P_bc = {'+': self.Γ_1_inv['+'] @ (self.R_1['+'].T * self.𝜔['+'].T) - (self.h['+'] / self.b['+'])**2 * self.Γ_1_inv['+'] @ self.Λ_1['+'] @ self.e_2.T @ self.Γ_2_inv['+'] @ (self.R_2['+'].T * self.𝛿['+'].T), 
                '-': self.Γ_1_inv['-'] @ (self.R_1['-'].T * self.𝜔['-'].T) - (self.h['-'] / self.b['-'])**2 * self.Γ_1_inv['-'] @ self.Λ_1['-'] @ self.e_2.T @ self.Γ_2_inv['-'] @ (self.R_2['-'].T * self.𝛿['-'].T)}
        
        if type(self.𝛾) == type(None):
            self.__get_𝛾()
        self.v_rbc = {'+': torch.sqrt((self.h['+'] / self.n) * torch.sum(torch.concat([self.e_0.T @ self.P_bc['+'] @ self.Σ['+'][0, :, :] @ self.P_bc['+'].T @ self.e_0] +\
                                                                         [self.𝛾[j]**2 * self.e_0.T @ self.P_bc['+'] @ self.Σ['+'][j + 1, :, :] @ self.P_bc['+'].T @ self.e_0 for j in range(self.q)]))),
                      '-': torch.sqrt((self.h['+'] / self.n) * torch.sum(torch.concat([self.e_0.T @ self.P_bc['-'] @ self.Σ['-'][0, :, :] @ self.P_bc['-'].T @ self.e_0] +\
                                                                         [self.𝛾[j]**2 * self.e_0.T @ self.P_bc['-'] @ self.Σ['-'][j + 1, :, :] @ self.P_bc['-'].T @ self.e_0 for j in range(self.q)])))}

    def __get_𝛾(self):
        self.Q = (self.Z.T * self.𝜔['-'].T) @ (self.I_n - self.R_1['-'] @ torch.linalg.inv((self.R_1['-'].T * self.𝜔['-'].T) @ self.R_1['-']) @ (self.R_1['-'].T * self.𝜔['-'].T)) @ self.W
        self.𝛾 = - (torch.linalg.inv(self.Q) @ (self.Z.T * self.𝜔['-'].T) @ (self.I_n - self.R_1['-'] @ torch.linalg.inv((self.R_1['-'].T * self.𝜔['-'].T) @ self.R_1['-']) @ (self.R_1['-'].T * self.𝜔['-'].T)) @ self.Y).flatten()
        
    def __build_edgeworth_terms(self):
        # Storing edgeworth terms as 2d vectors
        M, I, J = self.M, self.I, self.J
        
        self.ℓ_0_us = {'+': self.h['+'] * self.𝜔['+'] * torch.bmm((self.e_0.T @ self.Γ_1_inv['+']).expand(self.n, -1, -1), self.R_1['+'].unsqueeze(2)).squeeze(2),
                       '-': self.h['-'] * self.𝜔['-'] * torch.bmm((self.e_0.T @ self.Γ_1_inv['-']).expand(self.n, -1, -1), self.R_1['-'].unsqueeze(2)).squeeze(2)} # (n x 1)
        self.ℓ_0_bc = {'+': self.ℓ_0_us['+'] - self.b['+'] * (self.h['+']/self.b['+'])**2 * self.𝛿['+'] *\
                            torch.bmm((self.e_0.T @ self.Γ_1_inv['+'] @ self.Λ_1['+'] @ self.e_2.T @ self.Γ_2_inv['+']).expand(self.n, -1 , -1), self.R_2['+'].unsqueeze(2)).squeeze(2),
                       '-': self.ℓ_0_us['-'] - self.b['-'] * (self.h['-']/self.b['-'])**2 * self.𝛿['-'] *\
                            torch.bmm((self.e_0.T @ self.Γ_1_inv['-'] @ self.Λ_1['-'] @ self.e_2.T @ self.Γ_2_inv['-']).expand(self.n, -1 , -1), self.R_2['-'].unsqueeze(2)).squeeze(2)} # (n x 1)
        
        
        def build_ℓ_1_us(sn, I: torch.Tensor, J: torch.Tensor):
            M = I.shape[0]
            𝜔RRT = self.𝜔[sn].flatten()[J].view(-1, 1, 1) * torch.bmm(self.R_1[sn][J, :].unsqueeze(2), self.R_1[sn][J, :].unsqueeze(2).mT)
            term1 = torch.bmm((self.e_0.T @ self.Γ_1_inv[sn]).expand(M, -1, -1), self.Γ_1[sn].expand(M, -1, -1) - 𝜔RRT)
            term2 = torch.bmm(term1, self.Γ_1_inv[sn].expand(M, -1, -1))
            term3 = torch.bmm(term2, self.R_1[sn][J, :].unsqueeze(2))
            ℓ_1_us = self.h[sn]**2 * self.𝜔[sn][I, :] * term3.squeeze(2)
            return ℓ_1_us
        
        def build_ℓ_1_bc(sn, I: torch.Tensor, J: torch.Tensor):
            M = I.shape[0]
            ℓ_1_us = build_ℓ_1_us(sn, I, J)
            Dm, Ih = (self.D - self.cutoff), 1/self.h[sn]
            
            𝜔RRT = self.𝜔[sn].flatten()[J].view(-1, 1, 1) * torch.bmm(self.R_1[sn][J, :].unsqueeze(2), self.R_1[sn][J, :].unsqueeze(2).mT)
            term1a = self.h[sn] * torch.bmm(self.Γ_1[sn].expand(M, -1, -1) - 𝜔RRT, (self.Γ_1_inv[sn] @ self.Λ_1[sn] @ self.e_2.T).expand(M, -1, -1))
            
            𝜔RD2 = self.𝜔[sn].flatten()[J].view(-1, 1, 1) * torch.bmm(self.R_1[sn][J, :].unsqueeze(2), ((Dm * Ih)**2)[I, :].unsqueeze(2))
            E𝜔RD2 = torch.mean(𝜔RD2, dim = 0)
            term1b = self.h[sn] * torch.bmm(𝜔RD2 - E𝜔RD2.expand(M, -1, -1), self.e_2.T.expand(M, -1, -1))
            
            𝛿RRT = self.𝛿[sn].flatten()[J].view(-1, 1, 1) * torch.bmm(self.R_2[sn][J, :].unsqueeze(2), self.R_2[sn][J, :].unsqueeze(2).mT)
            term1c = self.b[sn] * torch.bmm((self.Λ_1[sn] @ self.e_2.T @ self.Γ_2_inv[sn]).expand(M, -1, -1), self.Γ_2[sn].expand(M, -1, -1) - 𝛿RRT)
            
            term1 = torch.bmm((self.e_0.T @ self.Γ_1_inv[sn]).expand(M, -1, -1), term1a + term1b + term1c)
            term2 = torch.bmm(term1, self.Γ_2_inv[sn].expand(M, -1, -1))
            term3 = torch.bmm(term2, self.R_2[sn][I, :].unsqueeze(2))
            
            ℓ_1_bc = ℓ_1_us - self.b[sn] * (self.h[sn]/self.b[sn])**2 * self.𝛿[sn][I, :] * term3.squeeze(2)
            return ℓ_1_bc
            
        self.ℓ_1_us = {'+': build_ℓ_1_us('+', I, J), '-': build_ℓ_1_us('-', I, J)} # (M x 1)
        self.ℓ_1_bc = {'+': build_ℓ_1_bc('+', I, J), '-': build_ℓ_1_bc('-', I, J)} # (M x 1)
        
        diag = torch.tensor(range(self.n), device = self.device, dtype = torch.int32)
        self.ℓ_1_us_diag = {'+': build_ℓ_1_us('+', diag, diag), '-': build_ℓ_1_us('-', diag, diag)} # (n x 1)
        self.ℓ_1_bc_diag = {'+': build_ℓ_1_bc('+', diag, diag), '-': build_ℓ_1_bc('-', diag, diag)} # (n x 1)

    def __get_q_1(self, sn: str, q: int, α = 0.05):
        M, I, J = self.M, self.I, self.J
        z = torch.tensor(norm.ppf(1 - α / 2), dtype=self.dtype, device=self.device)
        if q == 0:
            𝛾 = 1
        else:
            𝛾 = torch.abs(self.𝛾[q - 1])
        
        mean1 = torch.mean((𝛾 * self.ℓ_0_bc[sn] * self.ε[sn][:, [q]])**3)/self.b[sn]
        term1 = self.v_rbc[sn]**(-6) * mean1**2 * (z**3/3 + 7 * z / 4 + self.v_rbc[sn]**2 * z * (z**2 - 3)/4)
        
        mean2 = torch.mean(𝛾**2 * self.ℓ_0_bc[sn] * self.ℓ_1_bc_diag[sn] * self.ε[sn][:, [q]]**2)/self.b[sn]
        term2 = self.v_rbc[sn]**(-2) * mean2 * (-z * (z**2 - 3)/2)
        
        mean3 = torch.mean(𝛾**4 * self.ℓ_0_bc[sn]**4 * (self.ε[sn][:, [q]]**4 - self.σ[sn][:, [q]]**4))/self.b[sn]
        term3 = self.v_rbc[sn]**(-4) * mean3 * (z * (z**2 - 3)/8)
        
        mean4 = torch.mean(𝛾**2 * self.ℓ_0_bc[sn]**2 * self.𝛿[sn] * ((self.R_2[sn] @ self.Γ_2_inv[sn]) * self.R_2[sn]).sum(dim = 1, keepdim = True) * self.ε[sn][:, [q]]**2) 
        term4 = self.v_rbc[sn]**(-2) * mean4 * (z * (z**2 - 1)/2)
        
        mean5a = torch.mean(𝛾**3 * self.ℓ_0_bc[sn]**3 * self.R_2[sn] @ self.Γ_2_inv[sn] * self.ε[sn][:, [q]]**2, dim = 0) / self.b[sn]
        mean5b = torch.mean(𝛾 * self.ℓ_0_bc[sn] * self.𝛿[sn] * self.ε[sn][:, [q]]**2 * self.R_2[sn], dim = 0, keepdim = True).T
        term5 = self.v_rbc[sn]**(-4) * (mean5a @ mean5b)[0] * (z * (z**2 - 1))
        
        mean6 = torch.mean(𝛾**2 * self.ℓ_0_bc[sn]**2 * (self.𝛿[sn] * self.R_2[sn] @ self.Γ_2_inv[sn] * self.R_2[sn]).sum(dim = 1, keepdim = True)**2 * self.ε[sn][:, [q]]**2)
        term6 = self.v_rbc[sn]**(-2) * mean6 * (z * (z**2 - 1)/4)
        
        mean7a = torch.mean(𝛾 * self.ℓ_0_bc[sn] * self.ε[sn][:, [q]]**2 * self.𝛿[sn] * (self.R_2[sn] @ self.Γ_2_inv[sn]), dim = 0, keepdim = True)
        mean7b = torch.mean(𝛾**2 * (self.ℓ_0_bc[sn]**2).view(self.n, 1, 1) * torch.bmm(self.R_2[sn].unsqueeze(2), (self.R_2[sn] @ self.Γ_2_inv[sn]).unsqueeze(2).mT), dim = 0) / self.b[sn]
        mean7c = torch.mean(𝛾 * self.𝛿[sn] * self.R_2[sn] * self.ℓ_0_bc[sn] * self.ε[sn][:, [q]]**2, dim = 0, keepdim = True).T
        term7 = self.v_rbc[sn]**(-4) * (mean7a @ mean7b @ mean7c)[0, 0] * (z * (z**2 - 1)/2)
        
        mean8 = torch.mean(𝛾**4 * self.ℓ_0_bc[sn]**4 * self.ε[sn][:, [q]]**4)/self.b[sn]
        term8 = self.v_rbc[sn]**(-4) * mean8 * (-z * (z**2 - 3)/24)
        
        submean = torch.mean(𝛾**2 * self.ℓ_0_bc[sn]**2 * self.σ[sn][:, [q]]**2)
        mean9 = torch.mean((𝛾**2 * self.ℓ_0_bc[sn]**2 * self.σ[sn][:, [q]]**2 - submean.expand(self.n, 1)) * 𝛾**2 * self.ℓ_0_bc[sn]**2 * self.ε[sn][:, [q]]**2)/self.b[sn]
        term9 = self.v_rbc[sn]**(-4) * mean9 * (z * (z**2 - 1)/4)
        
        mean10 = torch.mean(𝛾**4 * self.ℓ_1_bc[sn] * self.ℓ_0_bc[sn][J, :]**2 * self.ℓ_0_bc[sn][I, :] * self.ε[sn][J, [q]]**2 * self.σ[sn][I, [q]]**2) / self.b[sn]**2
        term10 = self.v_rbc[sn]**(-4) * mean10 * (z * (z**2 - 3))
        
        mean11 = torch.mean(𝛾**2 * self.ℓ_1_bc[sn] * self.ℓ_0_bc[sn][I, :] * (𝛾**2 * self.ℓ_0_bc[sn][J, :]**2 * self.σ[sn][J, [q]]**2 - submean.expand(M, 1)) * self.ε[sn][I, [q]]**2) / self.b[sn]**2
        term11 = self.v_rbc[sn]**(-4) * mean11 * (-z)
        
        mean12 = torch.mean((𝛾**2 * self.ℓ_0_bc[sn][I, :]**2 * self.σ[sn][I, [q]]**2 - submean.expand(M, 1))**2) / self.b[sn]
        term12 = self.v_rbc[sn]**(-4) * mean12 * (-z * (z**2 + 1)/8)
        
        q_1 = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11 + term12
        return q_1
    
    def __get_q_2(self, sn: str, α=0.05):
        z = torch.tensor(norm.ppf(1 - α / 2), dtype=self.dtype, device=self.device)
        q_2 = - self.v_rbc[sn]**(-2) * z / 2
        return q_2
    
    def __get_q_3(self, sn: str, q: int, α=0.05):
        z = torch.tensor(norm.ppf(1 - α / 2), dtype=self.dtype, device=self.device)
        if q == 0:
            𝛾 = 1
        else:
            𝛾 = torch.abs(self.𝛾[q - 1])
        mean3 = torch.mean(𝛾**3 * self.ℓ_0_bc[sn]**3 * self.ε[sn][:, [q]]**3) / self.b[sn]
        q_3 = self.v_rbc[sn]**(-4) * mean3 * z**3 / 3
        return q_3
    
    def __get_𝜇_3(self, X, sn: str):
        𝛼_3 = (1/self.n) * self.e_3.T @ self.Γ_5_inv[sn] @ (self.R_5[sn].T * self.𝛿[sn].T) @ X
        return 𝛼_3[0, 0]
    
    def __get_𝜂_bc(self, sn: str, q: int):
        if q == 0:
            𝛾 = 1
        else:
            𝛾 = torch.abs(self.𝛾[q - 1])
        𝜂_bc = torch.sqrt(self.n * self.h[sn]) * self.h[sn]**3 * self.𝜇_3[sn][q] / factorial(3) * 𝛾 *\
            self.e_0.T @ self.Γ_1_inv[sn] @ (self.Λ_1_2[sn] - self.Λ_1[sn] @ self.e_2.T @ self.Γ_2_inv[sn] @ self.Λ_2_1[sn])
        return 𝜂_bc[0, 0]
    
    def __get_bandwidth(self, optim_mode = 'newton-cg', tol = 0.001):
        def obj(h):
            self.h['-'] = h[0]
            self.h['+'] = h[1]
            if torch.min(h) <= 0 or torch.max(h) > 10 * torch.max(torch.abs(self.D - self.cutoff)):
                return torch.inf
            self.b = {'+': 1/self.ρ * self.h['+'], '-': 1/self.ρ * self.h['-']}
            self.__build_matrices()
            self.__build_edgeworth_terms()
            self.𝜇_3 = {'+': torch.stack([self.__get_𝜇_3(self.Y, '+')] + [self.__get_𝜇_3(self.W[:, [j]], '+') for j in range(self.q)]), 
                   '-': torch.stack([self.__get_𝜇_3(self.Y, '-')] + [self.__get_𝜇_3(self.W[:, [j]], '-') for j in range(self.q)])}
            𝜂_bc = {'+': torch.stack([self.__get_𝜂_bc('+', j) for j in range(self.q + 1)]),
                    '-': torch.stack([self.__get_𝜂_bc('-', j) for j in range(self.q + 1)])}
            q_1 = {'+': torch.sum(torch.stack([self.__get_q_1('+', j) for j in range(self.q + 1)])), '-': torch.sum(torch.stack([self.__get_q_1('-', j) for j in range(self.q + 1)]))}
            q_2 = {'+': self.__get_q_2('+'), '-': self.__get_q_2('-')}
            q_3 = {'+': torch.stack([self.__get_q_3('+', j) for j in range(self.q + 1)]), '-': torch.stack([self.__get_q_3('-', j) for j in range(self.q + 1)])}
            loss = (( (1/(self.n * self.h['+'])) * q_1['+'] + self.n * self.h['+']**7 * torch.sum(𝜂_bc['+'])**2 * q_2['+'] +\
                      self.h['+']**3 * torch.sum(torch.stack([𝜂_bc['+'][j] * q_3['+'][j] for j in range(self.q + 1)])) )/self.n**(3/4))**2 +\
                   (( (1/(self.n * self.h['-'])) * q_1['-'] + self.n * self.h['-']**7 * torch.sum(𝜂_bc['-'])**2 * q_2['-'] +\
                      self.h['-']**3 * torch.sum(torch.stack([𝜂_bc['-'][j] * q_3['-'][j] for j in range(self.q + 1)])) )/self.n**(3/4))**2
            return loss
        
        h0 = torch.tensor([self.h['-'], self.h['+']])
        res = minimize(obj, h0, max_iter = 50, method = optim_mode, tol = tol)
        return res
        
    def fit(self):
        if type(self.custom_bandwidth) != type(None):
            self.h = {'-': self.custom_bandwidth[0], '+': self.custom_bandwidth[1]}
            status = True
        else:
            bres = self.__get_bandwidth()
            self.h = {'-': bres.x[0], '+': bres.x[1]}
            status = bres.success
        self.b = {'+': 1/self.ρ * self.h['+'], '-': 1/self.ρ * self.h['-']}
        self.__build_matrices()
        self.__get_𝛾()
        P_bc = self.P_bc['+'] - self.P_bc['-']
        est = torch.sum(torch.concat([(1/self.n) * self.e_0.T @ P_bc @ self.Y] + [(1/self.n) * self.𝛾[j] * self.e_0.T @ P_bc @ self.W[:, [j]] for j in range(self.q)]))
        se = torch.sqrt(self.v_rbc['+']**2/(self.n * self.h['+']) + self.v_rbc['-']**2/(self.n * self.h['-']))
        
        resid_pos = torch.concat([self.Y - self.R_2['+'] @ self.B_2β['+'][:, [0]]] + [self.W[:, [j]] - self.R_2['+'] @ self.B_2β['+'][:, [j + 1]] for j in range(self.q)], dim=1)
        resid_neg = torch.concat([self.Y - self.R_2['-'] @ self.B_2β['-'][:, [0]]] + [self.W[:, [j]] - self.R_2['-'] @ self.B_2β['-'][:, [j + 1]] for j in range(self.q)], dim=1)
        resids = (self.ind['+'] * resid_pos + self.ind['-'] * resid_neg).detach().cpu().numpy()
        def predict(d) -> np.ndarray:
            d = torch.as_tensor(d, dtype=self.dtype, device=self.device)
            if d.ndim == 1: d = d.reshape(-1, 1)
            Ih, dm, one_m = {'+': 1 / self.h['+'], '-': 1 / self.h['-']}, d - self.cutoff, torch.ones((d.shape[0], 1), dtype=self.dtype, device=self.device)
            ind = {'+': dm >= 0, '-': dm < 0}
            r = {'+': torch.cat([one_m, Ih['+'] * dm], dim=1),
                 '-': torch.cat([one_m, Ih['-'] * dm], dim=1)}
            Yhat = {'+': (1/self.n) * r['+'] @ self.P_bc['+'] @ self.Y, 
                    '-': (1/self.n) * r['+'] @ self.P_bc['-'] @ self.Y + torch.sum(torch.concat([(1/self.n) * self.𝛾[j] * self.e_0.T @ P_bc @ self.W[:, [j]] for j in range(self.q)]))}
            pred = ind['+'] * Yhat['+'] + ind['-'] * Yhat['-']
            return pred.flatten().detach().cpu().numpy()
            
        res = Results(model = 'Placebo Discontinuity Design',
                      est = est.item(),
                      se = se.item(),
                      resid = resids,
                      bandwidth = {'+': self.h['+'], '-': self.h['-']},
                      n = self.n,
                      predict = predict,
                      status = status)
        return res


class rdd:
    def __init__(self, Y: np.ndarray, D: np.ndarray, cutoff=0.0, alpha=0.05, kernel='triangle', 
                 bandwidth = None, dtype = torch.float32, device = 'cpu', seed = 10042002):
        self.dtype, self.device = dtype, device
        self.Y = torch.as_tensor(Y, dtype=dtype, device=device)
        if self.Y.ndim == 1: self.Y = self.Y.reshape(-1, 1)
        self.D = torch.as_tensor(D, dtype=dtype, device=device)
        if self.D.ndim == 1: self.D = self.D.reshape(-1, 1)
        self.n = int(self.D.shape[0])
        self.cutoff = torch.tensor(cutoff, dtype=dtype, device=device)
        self.alpha = torch.tensor(alpha, dtype=dtype, device=device)
        if kernel == 'triangle': 
            self.kernel = triangular_kernel
            self.ρ = 0.850
        elif kernel == 'rectangle': 
            self.kernel = rectangle_kernel
            self.ρ = 1
        else: 
            self.kernel = epanechnikov_kernel
            self.ρ = 0.898
        if type(bandwidth) != type(None):
            self.custom_bandwidth = torch.as_tensor(bandwidth, dtype=dtype, device=device).flatten()
        else:
            self.custom_bandwidth = None
        self.h = {'-': 3 * torch.std(self.D) * self.n**(-1/4), '+': 3 * torch.std(self.D) * self.n**(-1/4)}
        self.b = {'+': 1/self.ρ * self.h['+'], '-': 1/self.ρ * self.h['-']}
        self.gen = torch.Generator(device = device).manual_seed(seed)
        self.M = 2 * self.n * int(log(self.n))
        self.I, self.J = self.__sample_perms(self.n, self.M, self.device, self.gen)

    def __sample_perms(self, n: int, nsamples: int, device = 'cpu', gen = torch.Generator()) -> torch.Tensor:
        # random set of permutations when computing mean of edgeworth terms
        N = n * (n - 1)
        nsamples = min(nsamples, N)

        k = torch.randperm(N, device = device, dtype = torch.int32, generator = gen)[:nsamples] 
        I = k // (n - 1)
        jp = k % (n - 1)
        J = jp + (jp >= I).to(torch.int32)
        return I, J

    def __build_matrices(self):
        one_n = torch.ones((self.n, 1), dtype=self.dtype, device=self.device)
        Ih, Ib, Dm = {'+': 1 / self.h['+'], '-': 1 / self.h['-']}, {'+': 1 / self.b['+'], '-': 1 / self.b['-']}, self.D - self.cutoff
        self.R_1 = {'+': torch.cat([one_n, Ih['+'] * Dm], dim=1), '-': torch.cat([one_n, Ih['-'] * Dm], dim=1)}
        self.R_2 = {'+': torch.cat([one_n, Ib['+'] * Dm, (Ib['+'] * Dm)**2], dim=1), '-': torch.cat([one_n, Ib['-'] * Dm, (Ib['-'] * Dm)**2], dim=1)}

        self.ind = {'+': (self.D >= self.cutoff), '-': (self.D < self.cutoff)}
        self.𝜔 = {'+': (Ih['+'] * self.ind['+'] * self.kernel(Ih['+'] * Dm)), '-': (Ih['-'] * self.ind['-'] * self.kernel(Ih['-'] * Dm))}
        self.𝛿 = {'+': (Ib['+'] * self.ind['+'] * self.kernel(Ib['+'] * Dm)), '-': (Ib['-'] * self.ind['-'] * self.kernel(Ib['-'] * Dm))}
        
        self.I_n = torch.eye(self.n, dtype=self.dtype, device=self.device)
        self.Γ_1 = {'+': (1 / self.n) * (self.R_1['+'].T * self.𝜔['+'].T) @ self.R_1['+'], '-': (1 / self.n) * (self.R_1['-'].T * self.𝜔['-'].T) @ self.R_1['-']}
        self.Γ_2 = {'+': (1 / self.n) * (self.R_2['+'].T * self.𝛿['+'].T) @ self.R_2['+'], '-': (1 / self.n) * (self.R_2['-'].T * self.𝛿['-'].T) @ self.R_2['-']}
        self.Γ_1_inv = {'+': torch.linalg.pinv(self.Γ_1['+']), '-': torch.linalg.pinv(self.Γ_1['-'])}
        self.Γ_2_inv = {'+': torch.linalg.pinv(self.Γ_2['+']), '-': torch.linalg.pinv(self.Γ_2['-'])}
        self.Λ_1 = {'+': (1 / self.n) * (self.R_1['+'].T * self.𝜔['+'].T) @ (Ih['+'] * Dm)**2, '-': (1 / self.n) * (self.R_1['-'].T * self.𝜔['-'].T) @ (Ih['-'] * Dm)**2}
        self.Λ_2 = {'+': (1 / self.n) * (self.R_2['+'].T * self.𝛿['+'].T) @ (Ib['+'] * Dm)**2, '-': (1 / self.n) * (self.R_2['-'].T * self.𝛿['-'].T) @ (Ib['-'] * Dm)**2}
        self.Λ_1_2 = {'+': (1 / self.n) * (self.R_1['+'].T * self.𝜔['+'].T) @ (Ih['+'] * Dm)**3, '-': (1 / self.n) * (self.R_1['-'].T * self.𝜔['-'].T) @ (Ih['-'] * Dm)**3}
        self.Λ_2_1 = {'+': (1 / self.n) * (self.R_2['+'].T * self.𝛿['+'].T) @ (Ib['+'] * Dm)**2, '-': (1 / self.n) * (self.R_2['-'].T * self.𝛿['-'].T) @ (Ib['-'] * Dm)**2}
        self.e_0 = torch.tensor([[1.0], [0.0]], dtype=self.dtype, device=self.device)
        self.e_2 = torch.tensor([[0.0], [0.0], [1.0]], dtype=self.dtype, device=self.device)
        self.e_3 = torch.tensor([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0]], dtype=self.dtype, device=self.device)
        
        self.R_5 = {'+': torch.cat([torch.ones((self.n, 1), dtype=self.dtype, device=self.device), (Ih['+'] * Dm), (Ih['+'] * Dm)**2, (Ih['+'] * Dm)**3, (Ih['+'] * Dm)**4, (Ih['+'] * Dm)**5], dim=1),
                    '-': torch.cat([torch.ones((self.n, 1), dtype=self.dtype, device=self.device), (Ih['-'] * Dm), (Ih['-'] * Dm)**2, (Ih['-'] * Dm)**3, (Ih['-'] * Dm)**4, (Ih['-'] * Dm)**5], dim=1)}
        self.Γ_5 = {'+': (1 / self.n) * (self.R_5['+'].T * self.𝛿['+'].T) @ self.R_5['+'], '-': (1 / self.n) * (self.R_5['-'].T * self.𝛿['-'].T) @ self.R_5['-']}
        self.Γ_5_inv = {'+': torch.linalg.pinv(self.Γ_5['+']), '-': torch.linalg.pinv(self.Γ_5['-'])}

        self.B_2β = {'+': self.Γ_2_inv['+'] @ (self.R_2['+'].T * self.𝛿['+'].T) / self.n @ self.Y, 
                  '-': self.Γ_2_inv['-'] @ (self.R_2['-'].T * self.𝛿['-'].T) / self.n @ self.Y}
        self.H_1β = {'+': self.Γ_1_inv['+'] @ (self.R_1['+'].T * self.𝜔['+'].T) / self.n @ self.Y, 
                  '-': self.Γ_1_inv['-'] @ (self.R_1['-'].T * self.𝜔['-'].T) / self.n @ self.Y}
        self.ε = {'+': (self.Y - self.R_1['+'] @ self.H_1β['+']),  # (n, 1)
                  '-': (self.Y - self.R_1['-'] @ self.H_1β['-'])}  # (n, 1)
        self.σ = {'+': (self.Y - self.R_2['+'] @ self.B_2β['+']).abs(),  # (n, 1)
                  '-': (self.Y - self.R_2['+'] @ self.B_2β['+']).abs(),}  # (n, 1)
        self.Σ = {'+': torch.diag(self.σ['+'].flatten()**2),
                  '-': torch.diag(self.σ['-'].flatten()**2)}
        
        self.P_bc = {'+': self.Γ_1_inv['+'] @ (self.R_1['+'].T * self.𝜔['+'].T) - (self.h['+'] / self.b['+'])**2 * self.Γ_1_inv['+'] @ self.Λ_1['+'] @ self.e_2.T @ self.Γ_2_inv['+'] @ (self.R_2['+'].T * self.𝛿['+'].T), 
                     '-': self.Γ_1_inv['-'] @ (self.R_1['-'].T * self.𝜔['-'].T) - (self.h['-'] / self.b['-'])**2 * self.Γ_1_inv['-'] @ self.Λ_1['-'] @ self.e_2.T @ self.Γ_2_inv['-'] @ (self.R_2['-'].T * self.𝛿['-'].T)}
        self.v_rbc = {'+': torch.sqrt((self.h['+'] / self.n) * (self.e_0.T @ self.P_bc['+'] @ self.Σ['+'] @ self.P_bc['+'].T @ self.e_0)),
                      '-': torch.sqrt((self.h['-'] / self.n) * (self.e_0.T @ self.P_bc['-'] @ self.Σ['-'] @ self.P_bc['-'].T @ self.e_0))}

    def __build_edgeworth_terms(self):
        # Storing edgeworth terms as 2d vectors
        M, I, J = self.M, self.I, self.J
        
        self.ℓ_0_us = {'+': self.h['+'] * self.𝜔['+'] * torch.bmm((self.e_0.T @ self.Γ_1_inv['+']).expand(self.n, -1, -1), self.R_1['+'].unsqueeze(2)).squeeze(2),
                       '-': self.h['-'] * self.𝜔['-'] * torch.bmm((self.e_0.T @ self.Γ_1_inv['-']).expand(self.n, -1, -1), self.R_1['-'].unsqueeze(2)).squeeze(2)} # (n x 1)
        self.ℓ_0_bc = {'+': self.ℓ_0_us['+'] - self.b['+'] * (self.h['+']/self.b['+'])**2 * self.𝛿['+'] *\
                            torch.bmm((self.e_0.T @ self.Γ_1_inv['+'] @ self.Λ_1['+'] @ self.e_2.T @ self.Γ_2_inv['+']).expand(self.n, -1 , -1), self.R_2['+'].unsqueeze(2)).squeeze(2),
                       '-': self.ℓ_0_us['-'] - self.b['-'] * (self.h['-']/self.b['-'])**2 * self.𝛿['-'] *\
                            torch.bmm((self.e_0.T @ self.Γ_1_inv['-'] @ self.Λ_1['-'] @ self.e_2.T @ self.Γ_2_inv['-']).expand(self.n, -1 , -1), self.R_2['-'].unsqueeze(2)).squeeze(2)} # (n x 1)
        
        
        def build_ℓ_1_us(sn, I: torch.Tensor, J: torch.Tensor):
            M = I.shape[0]
            𝜔RRT = self.𝜔[sn].flatten()[J].view(-1, 1, 1) * torch.bmm(self.R_1[sn][J, :].unsqueeze(2), self.R_1[sn][J, :].unsqueeze(2).mT)
            term1 = torch.bmm((self.e_0.T @ self.Γ_1_inv[sn]).expand(M, -1, -1), self.Γ_1[sn].expand(M, -1, -1) - 𝜔RRT)
            term2 = torch.bmm(term1, self.Γ_1_inv[sn].expand(M, -1, -1))
            term3 = torch.bmm(term2, self.R_1[sn][J, :].unsqueeze(2))
            ℓ_1_us = self.h[sn]**2 * self.𝜔[sn][I, :] * term3.squeeze(2)
            return ℓ_1_us
        
        def build_ℓ_1_bc(sn, I: torch.Tensor, J: torch.Tensor):
            M = I.shape[0]
            ℓ_1_us = build_ℓ_1_us(sn, I, J)
            Dm, Ih = (self.D - self.cutoff), 1/self.h[sn]
            
            𝜔RRT = self.𝜔[sn].flatten()[J].view(-1, 1, 1) * torch.bmm(self.R_1[sn][J, :].unsqueeze(2), self.R_1[sn][J, :].unsqueeze(2).mT)
            term1a = self.h[sn] * torch.bmm(self.Γ_1[sn].expand(M, -1, -1) - 𝜔RRT, (self.Γ_1_inv[sn] @ self.Λ_1[sn] @ self.e_2.T).expand(M, -1, -1))
            
            𝜔RD2 = self.𝜔[sn].flatten()[J].view(-1, 1, 1) * torch.bmm(self.R_1[sn][J, :].unsqueeze(2), ((Dm * Ih)**2)[I, :].unsqueeze(2))
            E𝜔RD2 = torch.mean(𝜔RD2, dim = 0)
            term1b = self.h[sn] * torch.bmm(𝜔RD2 - E𝜔RD2.expand(M, -1, -1), self.e_2.T.expand(M, -1, -1))
            
            𝛿RRT = self.𝛿[sn].flatten()[J].view(-1, 1, 1) * torch.bmm(self.R_2[sn][J, :].unsqueeze(2), self.R_2[sn][J, :].unsqueeze(2).mT)
            term1c = self.b[sn] * torch.bmm((self.Λ_1[sn] @ self.e_2.T @ self.Γ_2_inv[sn]).expand(M, -1, -1), self.Γ_2[sn].expand(M, -1, -1) - 𝛿RRT)
            
            term1 = torch.bmm((self.e_0.T @ self.Γ_1_inv[sn]).expand(M, -1, -1), term1a + term1b + term1c)
            term2 = torch.bmm(term1, self.Γ_2_inv[sn].expand(M, -1, -1))
            term3 = torch.bmm(term2, self.R_2[sn][I, :].unsqueeze(2))
            
            ℓ_1_bc = ℓ_1_us - self.b[sn] * (self.h[sn]/self.b[sn])**2 * self.𝛿[sn][I, :] * term3.squeeze(2)
            return ℓ_1_bc
            
        self.ℓ_1_us = {'+': build_ℓ_1_us('+', I, J), '-': build_ℓ_1_us('-', I, J)} # (M x 1)
        self.ℓ_1_bc = {'+': build_ℓ_1_bc('+', I, J), '-': build_ℓ_1_bc('-', I, J)} # (M x 1)
        
        diag = torch.tensor(range(self.n), device = self.device, dtype = torch.int32)
        self.ℓ_1_us_diag = {'+': build_ℓ_1_us('+', diag, diag), '-': build_ℓ_1_us('-', diag, diag)} # (n x 1)
        self.ℓ_1_bc_diag = {'+': build_ℓ_1_bc('+', diag, diag), '-': build_ℓ_1_bc('-', diag, diag)} # (n x 1)

    def __get_q_1(self, sn: str, α = 0.05):
        M, I, J = self.M, self.I, self.J
        z = torch.tensor(norm.ppf(1 - α / 2), dtype=self.dtype, device=self.device)
        
        mean1 = torch.mean((self.ℓ_0_bc[sn] * self.ε[sn])**3)/self.b[sn]
        term1 = self.v_rbc[sn]**(-6) * mean1**2 * (z**3/3 + 7 * z / 4 + self.v_rbc[sn]**2 * z * (z**2 - 3)/4)
        
        mean2 = torch.mean(self.ℓ_0_bc[sn] * self.ℓ_1_bc_diag[sn] * self.ε[sn]**2)/self.b[sn]
        term2 = self.v_rbc[sn]**(-2) * mean2 * (-z * (z**2 - 3)/2)
        
        mean3 = torch.mean(self.ℓ_0_bc[sn]**4 * (self.ε[sn]**4 - self.σ[sn]**4))/self.b[sn]
        term3 = self.v_rbc[sn]**(-4) * mean3 * (z * (z**2 - 3)/8)
        
        mean4 = torch.mean(self.ℓ_0_bc[sn]**2 * self.𝛿[sn] * ((self.R_2[sn] @ self.Γ_2_inv[sn]) * self.R_2[sn]).sum(dim = 1, keepdim = True) * self.ε[sn]**2) 
        term4 = self.v_rbc[sn]**(-2) * mean4 * (z * (z**2 - 1)/2)
        
        mean5a = torch.mean(self.ℓ_0_bc[sn]**3 * self.R_2[sn] @ self.Γ_2_inv[sn] * self.ε[sn]**2, dim = 0) / self.b[sn]
        mean5b = torch.mean(self.ℓ_0_bc[sn] * self.𝛿[sn] * self.ε[sn]**2 * self.R_2[sn], dim = 0, keepdim = True).T
        term5 = self.v_rbc[sn]**(-4) * (mean5a @ mean5b)[0] * (z * (z**2 - 1))
        
        mean6 = torch.mean(self.ℓ_0_bc[sn]**2 * (self.𝛿[sn] * self.R_2[sn] @ self.Γ_2_inv[sn] * self.R_2[sn]).sum(dim = 1, keepdim = True)**2 * self.ε[sn]**2)
        term6 = self.v_rbc[sn]**(-2) * mean6 * (z * (z**2 - 1)/4)
        
        mean7a = torch.mean(self.ℓ_0_bc[sn] * self.ε[sn]**2 * self.𝛿[sn] * (self.R_2[sn] @ self.Γ_2_inv[sn]), dim = 0, keepdim = True)
        mean7b = torch.mean((self.ℓ_0_bc[sn]**2).view(self.n, 1, 1) * torch.bmm(self.R_2[sn].unsqueeze(2), (self.R_2[sn] @ self.Γ_2_inv[sn]).unsqueeze(2).mT), dim = 0) / self.b[sn]
        mean7c = torch.mean(self.𝛿[sn] * self.R_2[sn] * self.ℓ_0_bc[sn] * self.ε[sn]**2, dim = 0, keepdim = True).T
        term7 = self.v_rbc[sn]**(-4) * (mean7a @ mean7b @ mean7c)[0, 0] * (z * (z**2 - 1)/2)
        
        mean8 = torch.mean(self.ℓ_0_bc[sn]**4 * self.ε[sn]**4)/self.b[sn]
        term8 = self.v_rbc[sn]**(-4) * mean8 * (-z * (z**2 - 3)/24)
        
        submean = torch.mean(self.ℓ_0_bc[sn]**2 * self.σ[sn]**2)
        mean9 = torch.mean((self.ℓ_0_bc[sn]**2 * self.σ[sn]**2 - submean.expand(self.n, 1)) * self.ℓ_0_bc[sn]**2 * self.ε[sn]**2)/self.b[sn]
        term9 = self.v_rbc[sn]**(-4) * mean9 * (z * (z**2 - 1)/4)
        
        mean10 = torch.mean(self.ℓ_1_bc[sn] * self.ℓ_0_bc[sn][J, :]**2 * self.ℓ_0_bc[sn][I, :] * self.ε[sn][J, :]**2 * self.σ[sn][I, :]**2) / self.b[sn]**2
        term10 = self.v_rbc[sn]**(-4) * mean10 * (z * (z**2 - 3))
        
        mean11 = torch.mean(self.ℓ_1_bc[sn] * self.ℓ_0_bc[sn][I, :] * (self.ℓ_0_bc[sn][J, :]**2 * self.σ[sn][J, :]**2 - submean.expand(M, 1)) * self.ε[sn][I, :]**2) / self.b[sn]**2
        term11 = self.v_rbc[sn]**(-4) * mean11 * (-z)
        
        mean12 = torch.mean((self.ℓ_0_bc[sn][I, :]**2 * self.σ[sn][I, :]**2 - submean.expand(M, 1))**2) / self.b[sn]
        term12 = self.v_rbc[sn]**(-4) * mean12 * (-z * (z**2 + 1)/8)
        
        q_1 = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11 + term12
        return q_1
    
    def __get_q_2(self, sn: str, α=0.05):
        z = torch.tensor(norm.ppf(1 - α / 2), dtype=self.dtype, device=self.device)
        q_2 = - self.v_rbc[sn]**(-2) * z / 2
        return q_2
    
    def __get_q_3(self, sn: str, α=0.05):
        z = torch.tensor(norm.ppf(1 - α / 2), dtype=self.dtype, device=self.device)
        mean3 = torch.mean(self.ℓ_0_bc[sn]**3 * self.ε[sn]**3) / self.b[sn]
        q_3 = self.v_rbc[sn]**(-4) * mean3 * z**3 / 3
        return q_3
    
    def __get_𝜇_3(self, sn: str):
        𝛼_3 = (1/self.n) * self.e_3.T @ self.Γ_5_inv[sn] @ (self.R_5[sn].T * self.𝛿[sn].T) @ self.Y
        return 𝛼_3[0, 0]
    
    def __get_𝜂_bc(self, sn: str):
        𝜂_bc = torch.sqrt(self.n * self.h[sn]) * self.h[sn]**3 * self.𝜇_3[sn] / factorial(3) *\
            self.e_0.T @ self.Γ_1_inv[sn] @ (self.Λ_1_2[sn] - self.Λ_1[sn] @ self.e_2.T @ self.Γ_2_inv[sn] @ self.Λ_2_1[sn])
        return 𝜂_bc[0, 0]
    
    def __get_bandwidth(self, optim_mode = 'newton-cg', tol = 0.001):
        def obj(h):
            self.h['-'] = h[0]
            self.h['+'] = h[1]
            if torch.min(h) <= 0 or torch.max(h) > 10 * torch.max(torch.abs(self.D - self.cutoff)):
                return torch.inf
            self.b = {'+': 1/self.ρ * self.h['+'], '-': 1/self.ρ * self.h['-']}
            self.__build_matrices()
            self.__build_edgeworth_terms()
            self.𝜇_3 = {'+': self.__get_𝜇_3('+'), '-': self.__get_𝜇_3('-')}
            𝜂_bc = {'+': self.__get_η_bc('+'), '-': self.__get_η_bc('-')}
            q_1 = {'+': self.__get_q_1('+'), '-': self.__get_q_1('-')}
            q_2 = {'+': self.__get_q_2('+'), '-': self.__get_q_2('-')}
            q_3 = {'+': self.__get_q_3('+'), '-': self.__get_q_3('-')}
            loss = (( (1/(self.n * self.h['+'])) * q_1['+'] + self.n * self.h['+']**7 * 𝜂_bc['+']**2 * q_2['+'] + self.h['+']**3 * 𝜂_bc['+'] * q_3['+'] )/self.n**(3/4))**2 +\
                (( (1/(self.n * self.h['-'])) * q_1['-'] + self.n * self.h['-']**7 * 𝜂_bc['-']**2 * q_2['-'] + self.h['-']**3 * 𝜂_bc['-'] * q_3['-'] )/self.n**(3/4))**2
            return loss
        
        h0 = torch.tensor([self.h['-'], self.h['+']])
        res = minimize(obj, h0, max_iter = 50, method = optim_mode, tol = tol)
        return res
    
    def fit(self):
        if type(self.custom_bandwidth) != type(None):
            self.h = {'-': self.custom_bandwidth[0], '+': self.custom_bandwidth[1]}
            status = True
        else:
            bres = self.__get_bandwidth()
            self.h = {'-': bres.x[0], '+': bres.x[1]}
            status = bres.success
        self.b = {'+': 1/self.ρ * self.h['+'], '-': 1/self.ρ * self.h['-']}
        self.__build_matrices()
        P_bc = self.P_bc['+'] - self.P_bc['-']
        est = (1/self.n) * self.e_0.T @ P_bc @ self.Y
        se = torch.sqrt(self.v_rbc['+']**2/(self.n * self.h['+']) + self.v_rbc['-']**2/(self.n * self.h['-']))
        resid_pos = self.Y - self.R_2['+'] @ self.Γ_2_inv['+'] @ (self.R_2['+'].T * self.𝛿['+'].T) / self.n @ self.Y
        resid_neg = self.Y - self.R_2['-'] @ self.Γ_2_inv['-'] @ (self.R_2['-'].T * self.𝛿['-'].T) / self.n @ self.Y
        resids = (self.ind['+'] * resid_pos + self.ind['-'] * resid_neg).flatten().detach().cpu().numpy()
        def predict(d) -> np.ndarray:
            d = torch.as_tensor(d, dtype=self.dtype, device=self.device)
            if d.ndim == 1: d = d.reshape(-1, 1)
            Ih, dm, one_m = {'+': 1 / self.h['+'], '-': 1 / self.h['-']}, d - self.cutoff, torch.ones((d.shape[0], 1), dtype=self.dtype, device=self.device)
            ind = {'+': dm >= 0, '-': dm < 0}
            r = {'+': torch.cat([one_m, Ih['+'] * dm], dim=1),
                 '-': torch.cat([one_m, Ih['-'] * dm], dim=1)}
            Yhat = {'+': (1/self.n) * r['+'] @ self.P_bc['+'] @ self.Y, 
                    '-': (1/self.n) * r['-'] @ self.P_bc['-'] @ self.Y}
            pred = ind['+'] * Yhat['+'] + ind['-'] * Yhat['-']
            return pred.flatten().detach().cpu().numpy()
            
        res = Results(model = 'Regression Discontinuity Design',
                      est = est.item(),
                      se = se.item(),
                      resid = resids,
                      bandwidth = {'+': self.h['+'], '-': self.h['-']},
                      n = self.n,
                      predict = predict,
                      status = status)
        return res
    