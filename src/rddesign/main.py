import numpy as np, pandas as pd, warnings, torch, sys, os, re
from scipy.stats import norm
sys.path.append('/'.join(re.split('/|\\\\', os.path.dirname( __file__ ))[0:-1]))
from rddesign.helpers import *
from itertools import permutations
from torchmin import minimize, minimize_constr
from math import factorial, log

def rectangle_kernel(u: torch.Tensor) -> torch.Tensor:
    u = torch.as_tensor(u, dtype=torch.float32)
    a = torch.abs(u)
    return torch.where(a <= 1.0, 1.0, torch.zeros_like(a)).to(torch.float32)

def triangular_kernel(u: torch.Tensor) -> torch.Tensor:
    u = torch.as_tensor(u, dtype=torch.float32)
    a = torch.abs(u)
    return torch.where(a <= 1.0, 1.0 - a, torch.zeros_like(a)).to(torch.float32)

def epanechnikov_kernel(u: torch.Tensor) -> torch.Tensor:
    u = torch.as_tensor(u, dtype=torch.float32)
    a = torch.abs(u)
    return torch.where(a <= 1.0, (3/4) * (1 - a**2), torch.zeros_like(a)).to(torch.float32)

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
        self.cutoff = torch.tensor(cutoff, dtype=dtype, device=device)
        self.alpha = torch.tensor(alpha, dtype=dtype, device=device)
        if kernel == 'triangle': self.kernel = triangular_kernel
        elif kernel == 'rectangle': self.kernel = rectangle_kernel
        else: self.kernel = epanechnikov_kernel
        if type(bandwidth) != type(None):
            self.custom_bandwidth = torch.as_tensor(bandwidth, dtype=dtype, device=device).flatten()
        else:
            self.custom_bandwidth = None
        self.S = torch.cat([self.Y, self.W], dim=1)
        self.vecS = self.S.T.contiguous().reshape(-1, 1)
        self.vecW = self.W.T.contiguous().reshape(-1, 1)
        self.q = int(self.S.shape[1]) - 1
        self.h = {'+': torch.std(self.D) * self.n**(-1/5), '-': torch.std(self.D) * self.n**(-1/5)}
        self.b = {'+': 1/(0.778) * self.h['+'], '-': 1/(0.778) * self.h['-']}
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
    
    def __batch_kron(self, E: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        # E: (n, a, 1), L: (n, b, 1)
        # returns: (n, a*b, 1) corresponding to kron(E[i], L[i])
        K = torch.bmm(E, L.transpose(1, 2))          # (n, a, b)
        return K.reshape(E.shape[0], -1, 1)          # (n, a*b, 1))

    def __build_matrices(self):
        one_n = torch.ones((self.n, 1), dtype=self.dtype, device=self.device)
        Ih, Ib, Dm = {'+': 1 / self.h['+'], '-': 1 / self.h['-']}, {'+': 1 / self.b['+'], '-': 1 / self.b['-']}, self.D - self.cutoff
        self.R_1 = {'+': torch.cat([one_n, Ih['+'] * Dm], dim=1), '-': torch.cat([one_n, Ih['-'] * Dm], dim=1)}
        self.R_2 = {'+': torch.cat([one_n, Ib['+'] * Dm, (Ib['+'] * Dm)**2], dim=1), '-': torch.cat([one_n, Ib['-'] * Dm, (Ib['-'] * Dm)**2], dim=1)}

        self.ind = {'+': (self.D >= self.cutoff), '-': (self.D < self.cutoff)}
        self.𝜔 = {'+': (Ih['+'] * self.ind['+'] * self.kernel(Ih['+'] * Dm)), '-': (Ih['-'] * self.ind['-'] * self.kernel(Ih['-'] * Dm))}
        self.𝛿 = {'+': (Ib['+'] * self.ind['+'] * self.kernel(Ib['+'] * Dm)), '-': (Ib['-'] * self.ind['-'] * self.kernel(Ib['-'] * Dm))}
        self.K = {'+': torch.diag(self.𝜔['+'].flatten()), '-': torch.diag(self.𝜔['-'].flatten())}
        self.L = {'+': torch.diag(self.𝛿['+'].flatten()), '-': torch.diag(self.𝛿['-'].flatten())}
        
        self.I_n = torch.eye(self.n, dtype=self.dtype, device=self.device)
        self.Q = self.Z.T @ self.K['-'] @ (self.I_n - self.R_1['-'] @ torch.linalg.pinv(self.R_1['-'].T @ self.K['-'] @ self.R_1['-']) @ self.R_1['-'].T @ self.K['-'])  @ self.W
        self.Γ_1 = {'+': (1 / self.n) * (self.R_1['+'].T @ self.K['+'] @ self.R_1['+']), '-': (1 / self.n) * (self.R_1['-'].T @ self.K['-'] @ self.R_1['-'])}
        self.Γ_2 = {'+': (1 / self.n) * (self.R_2['+'].T @ self.L['+'] @ self.R_2['+']), '-': (1 / self.n) * (self.R_2['-'].T @ self.L['-'] @ self.R_2['-'])}
        self.Γ_1_inv = {'+': torch.linalg.pinv(self.Γ_1['+']), '-': torch.linalg.pinv(self.Γ_1['-'])}
        self.Γ_2_inv = {'+': torch.linalg.pinv(self.Γ_2['+']), '-': torch.linalg.pinv(self.Γ_2['-'])}
        self.Λ_1 = {'+': (1 / self.n) * (self.R_1['+'].T @ self.K['+'] @ (Ih['+'] * Dm)**2), '-': (1 / self.n) * (self.R_1['-'].T @ self.K['-'] @ (Ih['-'] * Dm)**2)}
        self.Λ_2 = {'+': (1 / self.n) * (self.R_2['+'].T @ self.L['+'] @ (Ib['+'] * Dm)**2), '-': (1 / self.n) * (self.R_2['-'].T @ self.L['-'] @ (Ib['-'] * Dm)**2)}
        self.Λ_1_2 = {'+': (1 / self.n) * (self.R_1['+'].T @ self.K['+'] @ (Ih['+'] * Dm)**3), '-': (1 / self.n) * (self.R_1['-'].T @ self.K['-'] @ (Ih['-'] * Dm)**3)}
        self.Λ_2_1 = {'+': (1 / self.n) * (self.R_2['+'].T @ self.L['+'] @ (Ib['+'] * Dm)**2), '-': (1 / self.n) * (self.R_2['-'].T @ self.L['-'] @ (Ib['-'] * Dm)**2)}
        self.e_2 = torch.tensor([[0.0], [0.0], [1.0]], dtype=self.dtype, device=self.device)
        self.l_q1 = torch.ones((self.q + 1, 1), dtype=self.dtype, device=self.device)
        self.e_3 = torch.tensor([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0]], dtype=self.dtype, device=self.device)
        self.I_q1 = torch.eye(self.q + 1, dtype=self.dtype, device=self.device)
        
        self.R_5 = {'+': torch.cat([torch.ones((self.n, 1), dtype=self.dtype, device=self.device), (Ih['+'] * Dm), (Ih['+'] * Dm)**2, (Ih['+'] * Dm)**3, (Ih['+'] * Dm)**4, (Ih['+'] * Dm)**5], dim=1),
                    '-': torch.cat([torch.ones((self.n, 1), dtype=self.dtype, device=self.device), (Ih['-'] * Dm), (Ih['-'] * Dm)**2, (Ih['-'] * Dm)**3, (Ih['-'] * Dm)**4, (Ih['-'] * Dm)**5], dim=1)}
        self.Γ_5 = {'+': (1 / self.n) * (self.R_5['+'].T @ self.K['+'] @ self.R_5['+']), '-': (1 / self.n) * (self.R_5['-'].T @ self.K['-'] @ self.R_5['-'])}
        self.Γ_5_inv = {'+': torch.linalg.pinv(self.Γ_5['+']), '-': torch.linalg.pinv(self.Γ_5['-'])}
        self.𝛾 = (torch.linalg.pinv(self.Q) @ self.Z.T @ self.K['-'] @ (self.I_n - self.R_1['-'] @ torch.linalg.pinv(self.R_1['-'].T @ self.K['-'] @ self.R_1['-']) @ self.R_1['-'].T @ self.K['-']) @ self.Y).flatten()
        self.s = torch.cat([torch.tensor([[1.0, 0.0]], dtype=self.dtype, device=self.device), torch.stack([torch.tensor([-𝛾.item(), 0.0], dtype=self.dtype, device=self.device) for 𝛾 in self.𝛾])], dim=0).flatten().reshape(-1, 1)

        vecB2β = {'+': torch.kron(self.I_q1, self.Γ_2_inv['+'] @ self.R_2['+'].T @ self.L['+'] / self.n) @ self.vecS, 
                  '-': torch.kron(self.I_q1, self.Γ_2_inv['-'] @ self.R_2['-'].T @ self.L['-'] / self.n) @ self.vecS}
        self.ε = {'+': torch.stack([self.S[i, None].T - torch.kron(self.I_q1, self.R_2['+'][i, None]) @ vecB2β['+'] for i in range(self.n)], dim=0),  # (n, q+1, 1)
                  '-': torch.stack([self.S[i, None].T - torch.kron(self.I_q1, self.R_2['-'][i, None]) @ vecB2β['-'] for i in range(self.n)], dim=0),}  # (n, q+1, 1)
        self.σ = {'+': self.ε['+'].abs(),  # (n, q+1, 1)
                  '-': self.ε['-'].abs(),}  # (n, q+1, 1)
        self.Σ = {'+': torch.diag((self.σ['+'][:, :, 0] ** 2).transpose(0, 1).contiguous().flatten()),
                  '-': torch.diag((self.σ['-'][:, :, 0] ** 2).transpose(0, 1).contiguous().flatten()),}
        self.P_bc = {'+': self.Γ_1_inv['+'] @ self.R_1['+'].T @ self.K['+'] - (self.h['+'] / self.b['+'])**2 * self.Γ_1_inv['+'] @ self.Λ_1['+'] @ self.e_2.T @ self.Γ_2_inv['+'] @ self.R_2['+'].T @ self.L['+'], 
                '-': self.Γ_1_inv['-'] @ self.R_1['-'].T @ self.K['-'] - (self.h['-'] / self.b['-'])**2 * self.Γ_1_inv['-'] @ self.Λ_1['-'] @ self.e_2.T @ self.Γ_2_inv['-'] @ self.R_2['-'].T @ self.L['-']}
        self.v_bc = torch.sqrt((self.h['+'] / self.n) * (self.s.T @ torch.kron(self.I_q1, self.P_bc['+']) @ self.Σ['+'] @ torch.kron(self.I_q1, self.P_bc['+']).T @ self.s) +\
                                (self.h['-'] / self.n) * (self.s.T @ torch.kron(self.I_q1, self.P_bc['-']) @ self.Σ['-'] @ torch.kron(self.I_q1, self.P_bc['-']).T @ self.s))

    def __build_edgeworth_terms(self):
        # Storing edgeworth terms as 3d tensors
        M, I, J = self.M, self.I, self.J
        
        self.ℓ_0_us = {'+': self.h['+'] * self.𝜔['+'].flatten().view(self.n, 1, 1) * torch.bmm(self.Γ_1_inv['+'].unsqueeze(0).expand(self.n, -1, -1), self.R_1['+'].unsqueeze(2)),
                       '-': self.h['-'] * self.𝜔['-'].flatten().view(self.n, 1, 1) * torch.bmm(self.Γ_1_inv['-'].unsqueeze(0).expand(self.n, -1, -1), self.R_1['-'].unsqueeze(2))}
        self.ℓ_0_bc = {'+': self.ℓ_0_us['+'] - (self.b['+'] * (self.h['+']/self.b['+'])**2 * self.𝛿['+'].flatten()).view(self.n, 1, 1) *\
                            torch.bmm((self.Γ_1_inv['+'] @ self.Λ_1['+'] @ self.e_2.T @ self.Γ_2_inv['+']).unsqueeze(0).expand(self.n, -1 , -1), self.R_2['+'].unsqueeze(2)),
                       '-': self.ℓ_0_us['-'] - (self.b['-'] * (self.h['-']/self.b['-'])**2 * self.𝛿['-'].flatten()).view(self.n, 1, 1) *\
                            torch.bmm((self.Γ_1_inv['-'] @ self.Λ_1['-'] @ self.e_2.T @ self.Γ_2_inv['-']).unsqueeze(0).expand(self.n, -1 , -1), self.R_2['-'].unsqueeze(2))}
        
        def build_ℓ_1_us(sn, I: torch.Tensor, J: torch.Tensor):
            outer = self.R_1[sn][I].unsqueeze(2) * self.R_1[sn][J].unsqueeze(1) # (M, 2, 2)
            diff = self.Γ_1[sn].unsqueeze(0) - self.𝜔[sn].flatten()[J].view(-1, 1, 1) * outer # (M, 2, 2)  (Γ_1 - 𝜔 * self.r_j @ self.R_j.T)
            # batched: Γ_inv @ diff @ Γ_inv @ Ri^T
            tmp = torch.bmm(self.Γ_1_inv[sn].unsqueeze(0).expand(I.shape[0], -1, -1), diff)
            tmp = torch.bmm(tmp, self.Γ_1_inv[sn].unsqueeze(0).expand(I.shape[0], -1, -1))
            tmp = torch.bmm(tmp, self.R_1[sn][I].unsqueeze(2))
            ℓ_1_us = (self.h[sn])**2 * self.𝜔[sn].flatten()[I].view(-1,1,1) * tmp  # (M, 2, 1)
            return ℓ_1_us
        
        def build_ℓ_1_bc(sn, I: torch.Tensor, J: torch.Tensor):
            X = ((self.D - self.cutoff) / self.h[sn]).flatten()**2
            𝜔RD = self.𝜔[sn].flatten()[I].unsqueeze(1) * self.R_1[sn][J] * X[I].unsqueeze(1)  # (M, 2)
            E𝜔RD = 𝜔RD.mean(dim=0, keepdim=True)
            ℓ_1_us = build_ℓ_1_us(sn, I, J)
            
            # ---- Build chunky (M, 2, 3) term ----
            outer = self.R_1[sn][J].unsqueeze(2) * self.R_1[sn][J].unsqueeze(1)                                         # (M, 2, 2)
            diff = self.h[sn] * (self.Γ_1[sn].unsqueeze(0) - self.𝜔[sn].flatten()[J].view(-1, 1, 1) * outer) # (M, 2, 2)
            tmpA = torch.bmm(diff, self.Γ_1_inv[sn].unsqueeze(0).expand(I.shape[0], -1, -1))                                     # (M, 2, 2)
            tmpA = torch.bmm(tmpA, self.Λ_1[sn].unsqueeze(0).expand(I.shape[0], -1, -1))                                         # (M, 2, 1)
            A = torch.bmm(tmpA, self.e_2.T.unsqueeze(0).expand(I.shape[0], -1, -1)) # (M, 2, 3) h * (Γ_1 - 𝜔_j * r_j @ r_j.T) @ Γ_1^(-1) @ Λ_1 @ e_2.T
            
            tmpB = self.𝜔[sn].flatten()[I].unsqueeze(1) * self.R_1[sn][J] * X[I].unsqueeze(1) - E𝜔RD                   # (M, 2)
            B = torch.bmm((self.h[sn] * tmpB).unsqueeze(2),  self.e_2.T.unsqueeze(0).expand(I.shape[0], -1, -1)) # (M, 2, 3) h * (𝜔_i * r_j * x_i - EωRD) @ e_2.T
            
            outer2 = self.R_2[sn][J].unsqueeze(2) * self.R_2[sn][J].unsqueeze(1)                                        # (M, 3, 3)
            diff2 = self.Γ_2[sn].unsqueeze(0) -  self.𝛿[sn].flatten()[J].view(-1, 1, 1) * outer2                        # (M, 3, 3)
            tmpC = torch.bmm((self.e_2.T @ self.Γ_2_inv[sn]).unsqueeze(0).expand(I.shape[0], -1, -1), diff2)                     # (M, 1, 3)
            C = self.b[sn] * torch.bmm(self.Λ_1[sn].unsqueeze(0).expand(I.shape[0], -1, -1), tmpC)   # (M, 2, 3) b * Λ_1 * e_2.T * Γ_2^(-1) * (Γ_2 - 𝛿_j * r2_j @ r2_j.T)
            
            tmp = torch.bmm(self.Γ_1_inv[sn].unsqueeze(0).expand(I.shape[0], -1, -1), A + B + C)                                 # (M, 2, 3)
            v = torch.bmm(self.Γ_2_inv[sn].unsqueeze(0).expand(I.shape[0], -1, -1), self.R_2[sn][I].unsqueeze(2))                # (M, 3, 1)
            tmp = torch.bmm(tmp, v)                                                                                     # (M, 2, 1)
            ℓ_1_bc = ℓ_1_us - (self.b[sn] * (self.h[sn]/self.b[sn])**2 * self.𝛿[sn].flatten()[I]).view(I.shape[0], 1, 1) * tmp   # (M, 2, 1)
            return ℓ_1_bc
            
        self.ℓ_1_us = {'+': build_ℓ_1_us('+', I, J), '-': build_ℓ_1_us('-', I, J)}
        self.ℓ_1_bc = {'+': build_ℓ_1_bc('+', I, J), '-': build_ℓ_1_bc('-', I, J)}
        
        diag = torch.tensor(range(self.n), device = self.device, dtype = torch.int32)
        self.ℓ_1_us_diag = {'+': build_ℓ_1_us('+', diag, diag), '-': build_ℓ_1_us('-', diag, diag)}
        self.ℓ_1_bc_diag = {'+': build_ℓ_1_bc('+', diag, diag), '-': build_ℓ_1_bc('-', diag, diag)}

    def __get_q_1(self, sn: str, α=0.05):
        z = torch.tensor(norm.ppf(1 - α / 2), dtype=self.dtype, device=self.device)
        def sT_kron_batch(E, L):
            K = self.__batch_kron(E, L)                           # (N, a*b, 1)
            sT = self.s.T.unsqueeze(0).expand(K.shape[0], -1, -1) # (N, 1, a*b)
            return torch.bmm(sT, K).squeeze(-1).squeeze(-1)       # (N,)
        
        l_q1 = self.l_q1.unsqueeze(0).expand(self.n, -1, -1)  # (n, q+1, 1)

        # -------- term1..term4, term6, term8, term9, term12 (all i-only) --------
        mean1 = ((sT_kron_batch(self.ε[sn], self.ℓ_0_bc[sn])**3) / self.h[sn]).mean()
        term1 = (self.v_bc**(-6) * (mean1**2) *
                (z**3 / 3 + 7*z/4 + self.v_bc**2 * z * (z**2 - 3)/4))
        mean2 = ((sT_kron_batch(self.ε[sn], self.ℓ_0_bc[sn]) * sT_kron_batch(self.ε[sn], self.ℓ_1_bc_diag[sn])) / self.h[sn]).mean()
        term2 = self.v_bc**(-2) * mean2 * (-(z**2 - 3) / 4)

        mean3 = (((sT_kron_batch(self.ε[sn], self.ℓ_0_bc[sn])**4) - (sT_kron_batch(self.σ[sn], self.ℓ_0_bc[sn])**4)) / self.h[sn]).mean()
        term3 = self.v_bc**(-4) * mean3 * (z * (z**2 - 3) / 8)

        mean4 = ((sT_kron_batch(self.ε[sn], self.ℓ_0_bc[sn])**2) / self.h[sn] * (self.b[sn] * self.𝛿[sn].flatten()) * (self.R_2[sn] @ self.Γ_2_inv[sn] * self.R_2[sn]).sum(dim=1)).mean()
        term4 = self.v_bc**(-2) * mean4 * (z * (z**2 - 1) / 2)

        mean6 = (((sT_kron_batch(self.ε[sn], self.ℓ_0_bc[sn])**2) / (self.h[sn]**2)) * (self.b[sn] * self.𝛿[sn].flatten() * (self.R_2[sn] @ self.Γ_2_inv[sn] * self.R_2[sn]).sum(dim=1))**2).mean()
        term6 = self.v_bc**(-2) * mean6 * (z * (z**2 - 1)) / 4

        mean8 = ((sT_kron_batch(self.ε[sn], self.ℓ_0_bc[sn])**4) / self.h[sn]).mean()
        term8 = self.v_bc**(-4) * mean8 * (-z * (z**2 - 3) / 24)

        submean = (sT_kron_batch(self.σ[sn], self.ℓ_0_bc[sn])**2).mean()  # scalar
        mean9 = (((sT_kron_batch(self.σ[sn], self.ℓ_0_bc[sn])**2 - submean) * (sT_kron_batch(self.ε[sn], self.ℓ_0_bc[sn])**2)) / self.h[sn]).mean()
        term9 = self.v_bc**(-4) * mean9 * (z * (z**2 - 1) / 4)

        mean12 = ((sT_kron_batch(self.σ[sn], self.ℓ_0_bc[sn])**2 - submean) / self.h[sn]).mean()
        term12 = self.v_bc**(-4) * mean12 * (-z * (z**2 + 1) / 8)

        # -------- term5, term7 --------
        meana = ((((sT_kron_batch(self.ε[sn], self.ℓ_0_bc[sn])**2) / self.h[sn]) * sT_kron_batch(l_q1, self.ℓ_0_bc[sn])).unsqueeze(1) * self.R_2[sn] @ self.Γ_2_inv[sn]).mean(dim=0, keepdim=True)  # (1,3)
        meanb = (((self.b[sn] / self.h[sn]) * self.𝛿[sn].flatten() * sT_kron_batch(self.ε[sn] * self.ε[sn], self.ℓ_0_bc[sn])).unsqueeze(1) * self.R_2[sn]).mean(dim=0).unsqueeze(1)    # (3,1)
        term5 = (self.v_bc**(-4) * (meana @ meanb).squeeze() * (z * (z**2 - 1)))

        mean7a = ((((sT_kron_batch(self.ε[sn] * self.ε[sn], self.ℓ_0_bc[sn])) / self.h[sn]) * (self.b[sn] * self.𝛿[sn].flatten())).unsqueeze(1) * self.R_2[sn] @ self.Γ_2_inv[sn]).mean(dim=0, keepdim=True)    # (1,3)
        outer2 = self.R_2[sn].unsqueeze(2) * self.R_2[sn].unsqueeze(1)                 # (n,3,3) = R2_i^T R2_i
        mean7b = ((sT_kron_batch(l_q1, self.ℓ_0_bc[sn])**2) / self.h[sn]).view(self.n,1,1) * torch.matmul(outer2, self.Γ_2_inv[sn]) # (n,3,3)
        mean7b = mean7b.mean(dim=0)                                        # (3,3)
        mean7c = ((self.b[sn] / self.h[sn]) * self.𝛿[sn].flatten() * sT_kron_batch(self.ε[sn] * self.ε[sn], self.ℓ_0_bc[sn])).unsqueeze(1) * self.R_2[sn]           # (n,3)
        mean7c = mean7c.mean(dim=0).unsqueeze(1)                           # (3,1)
        term7 = (self.v_bc**(-4) * (mean7a @ mean7b @ mean7c).squeeze() * (z * (z**2 - 1) / 2))

        # -------- term10, term11 over sampled ordered pairs (I,J) --------
        M, I, J = self.M, self.I, self.J
        mean10 = (((sT_kron_batch(self.σ[sn][I], self.ℓ_1_bc[sn])  * sT_kron_batch(self.σ[sn], self.ℓ_0_bc[sn])[I] * (sT_kron_batch(self.ε[sn], self.ℓ_0_bc[sn])[I]**2)) / (self.h[sn]**2)).mean())
        term10 = self.v_bc**(-4) * mean10 * (z * (z**2 - 3))

        mean11 = (((sT_kron_batch(self.ε[sn], self.ℓ_0_bc[sn])[I] * sT_kron_batch(self.ε[sn][I], self.ℓ_1_bc[sn]) * (sT_kron_batch(self.σ[sn], self.ℓ_0_bc[sn])[I]**2 - submean)) / (self.h[sn]**2)).mean())
        term11 = self.v_bc**(-4) * mean11 * (z * (z**2 - 3))

        q_1 = term1 + term2 + term3 - term4 - term5 + term6 + term7 + term8 + term9 + term10 + term11 + term12
        return q_1

    def __get_q_2(self, α=0.05):
        z = torch.tensor(norm.ppf(1 - α / 2), dtype=self.dtype, device=self.device)
        q_2 = - self.v_bc**(-2) * z / 2
        return q_2
    
    def __get_q_3(self, sn: str, α=0.05):
        z = torch.tensor(norm.ppf(1 - α / 2), dtype=self.dtype, device=self.device)
        mean3 = ((torch.bmm(self.s.T.unsqueeze(0).expand(self.n, -1, -1), self.__batch_kron(self.ε[sn], self.ℓ_0_bc[sn])).squeeze(-1).squeeze(-1)**3) / self.h[sn]).mean() 
        q_3 = - self.v_bc**(-4) * mean3 * z**3 / 3
        return q_3
    
    def __get_𝜇_3(self, sn: str):
        𝛼_3 = torch.kron(self.I_q1, (1/self.n) * self.e_3.T @ self.Γ_5_inv[sn] @ self.R_5[sn].T @ self.K[sn]) @ self.vecS # R^{(q + 1)}
        return 𝛼_3
    
    def __get_bandwidth(self, optim_mode = 'newton-cg', tol = 0.001):
        def obj(h):
            self.h['-'] = h[0]
            self.h['+'] = h[1]
            if torch.min(h) <= 0 or torch.max(h) > 10 * torch.max(torch.abs(self.D - self.cutoff)):
                return torch.inf
            self.b = {'+': 1/(0.778) * self.h['+'], '-': 1/(0.778) * self.h['-']}
            self.__build_matrices()
            self.__build_edgeworth_terms()
            𝜇_3 = {'+': self.__get_𝜇_3('+'), '-': self.__get_𝜇_3('-')}
            𝜂_bc = {'+': torch.sqrt(self.n * self.h['+']) * self.h['+']**3 * self.s.T @ torch.kron(𝜇_3['+'] / factorial(3), self.Γ_1_inv['+'] @ (self.Λ_1_2['+'] - self.Λ_1['+'] @ self.e_2.T @ self.Γ_2_inv['+'] @ self.Λ_2_1['+'])),
                    '-': torch.sqrt(self.n * self.h['-']) * self.h['-']**3 * self.s.T @ torch.kron(𝜇_3['-'] / factorial(3), self.Γ_1_inv['-'] @ (self.Λ_1_2['-'] - self.Λ_1['-'] @ self.e_2.T @ self.Γ_2_inv['-'] @ self.Λ_2_1['-']))}
            q_1 = {'+': self.__get_q_1('+'), '-': self.__get_q_1('-')}
            q_2 = self.__get_q_2()
            q_3 = {'+': self.__get_q_3('+'), '-': self.__get_q_3('-')}
            loss = (( (1/(self.n * self.h['+'])) * q_1['+'] + self.n * self.h['+']**7 * 𝜂_bc['+']**2 * q_2 + self.h['+']**3 * 𝜂_bc['+'] * q_3['+'] )/self.n**(3/4))**2 +\
                (( (1/(self.n * self.h['-'])) * q_1['-'] + self.n * self.h['-']**7 * 𝜂_bc['-']**2 * q_2 + self.h['-']**3 * 𝜂_bc['-'] * q_3['-'] )/self.n**(3/4))**2
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
        self.b = {'+': 1/(0.778) * self.h['+'], '-': 1/(0.778) * self.h['-']}
        self.__build_matrices()
        P_bc = self.P_bc['+'] - self.P_bc['-']
        est = (1/self.n) * self.s.T @ torch.kron(self.I_q1, P_bc) @ self.vecS
        se = torch.sqrt(( 1 / self.n**2) * (self.s.T @ torch.kron(self.I_q1, self.P_bc['+']) @ self.Σ['+'] @ torch.kron(self.I_q1, self.P_bc['+']).T @ self.s) +\
                        (1 / self.n**2) * (self.s.T @ torch.kron(self.I_q1, self.P_bc['-']) @ self.Σ['-'] @ torch.kron(self.I_q1, self.P_bc['-']).T @ self.s))
        #se = self.__bootstrap_se()
        resid_pos = self.Y - torch.hstack([self.R_1['+'], self.W]) @ torch.linalg.inv(torch.vstack([self.R_1['+'].T, self.Z.T]) @ self.K['+'] @ torch.hstack([self.R_1['+'], self.W])) @ (torch.vstack([self.R_1['+'].T, self.Z.T]) @ self.K['+'] @ self.Y)
        resid_neg = self.Y - torch.hstack([self.R_1['-'], self.W]) @ torch.linalg.inv(torch.vstack([self.R_1['-'].T, self.Z.T]) @ self.K['-'] @ torch.hstack([self.R_1['-'], self.W])) @ (torch.vstack([self.R_1['-'].T, self.Z.T]) @ self.K['-'] @ self.Y)
        resids = (self.ind['+'] * resid_pos + self.ind['-'] * resid_neg).flatten().detach().cpu().numpy()
        
        def predict(d) -> np.ndarray:
            d = torch.as_tensor(d, dtype=self.dtype, device=self.device)
            if d.ndim == 1: d = d.reshape(-1, 1)
            Ih, dm, one_m = {'+': 1 / self.h['+'], '-': 1 / self.h['-']}, d - self.cutoff, torch.ones((d.shape[0], 1), dtype=self.dtype, device=self.device)
            ind = {'+': dm >= 0, '-': dm < 0}
            sw = self.s[2:, :]
            r = {'+': torch.cat([one_m, Ih['+'] * dm], dim=1),
                 '-': torch.cat([one_m, Ih['-'] * dm], dim=1)}
            Yhat = {'+': (1/self.n) * r['+'] @ self.P_bc['+'] @ self.Y, 
                    '-': (1/self.n) * r['-'] @ self.P_bc['-'] @ self.Y - (1/self.n) * sw.T @ torch.kron(torch.eye(self.q, dtype=self.dtype, device=self.device), P_bc) @ self.vecW}
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
        self.h = {'-': 2 * torch.std(self.D) * self.n**(-1/4), '+': 2 * torch.std(self.D) * self.n**(-1/4)}
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
        self.K = {'+': torch.diag(self.𝜔['+'].flatten()), '-': torch.diag(self.𝜔['-'].flatten())}
        self.L = {'+': torch.diag(self.𝛿['+'].flatten()), '-': torch.diag(self.𝛿['-'].flatten())}
        
        self.I_n = torch.eye(self.n, dtype=self.dtype, device=self.device)
        self.Γ_1 = {'+': (1 / self.n) * (self.R_1['+'].T @ self.K['+'] @ self.R_1['+']), '-': (1 / self.n) * (self.R_1['-'].T @ self.K['-'] @ self.R_1['-'])}
        self.Γ_2 = {'+': (1 / self.n) * (self.R_2['+'].T @ self.L['+'] @ self.R_2['+']), '-': (1 / self.n) * (self.R_2['-'].T @ self.L['-'] @ self.R_2['-'])}
        self.Γ_1_inv = {'+': torch.linalg.pinv(self.Γ_1['+']), '-': torch.linalg.pinv(self.Γ_1['-'])}
        self.Γ_2_inv = {'+': torch.linalg.pinv(self.Γ_2['+']), '-': torch.linalg.pinv(self.Γ_2['-'])}
        self.Λ_1 = {'+': (1 / self.n) * (self.R_1['+'].T @ self.K['+'] @ (Ih['+'] * Dm)**2), '-': (1 / self.n) * (self.R_1['-'].T @ self.K['-'] @ (Ih['-'] * Dm)**2)}
        self.Λ_2 = {'+': (1 / self.n) * (self.R_2['+'].T @ self.L['+'] @ (Ib['+'] * Dm)**2), '-': (1 / self.n) * (self.R_2['-'].T @ self.L['-'] @ (Ib['-'] * Dm)**2)}
        self.Λ_1_2 = {'+': (1 / self.n) * (self.R_1['+'].T @ self.K['+'] @ (Ih['+'] * Dm)**3), '-': (1 / self.n) * (self.R_1['-'].T @ self.K['-'] @ (Ih['-'] * Dm)**3)}
        self.Λ_2_1 = {'+': (1 / self.n) * (self.R_2['+'].T @ self.L['+'] @ (Ib['+'] * Dm)**2), '-': (1 / self.n) * (self.R_2['-'].T @ self.L['-'] @ (Ib['-'] * Dm)**2)}
        self.e_0 = torch.tensor([[1.0], [0.0]], dtype=self.dtype, device=self.device)
        self.e_2 = torch.tensor([[0.0], [0.0], [1.0]], dtype=self.dtype, device=self.device)
        self.e_3 = torch.tensor([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0]], dtype=self.dtype, device=self.device)
        
        self.R_5 = {'+': torch.cat([torch.ones((self.n, 1), dtype=self.dtype, device=self.device), (Ih['+'] * Dm), (Ih['+'] * Dm)**2, (Ih['+'] * Dm)**3, (Ih['+'] * Dm)**4, (Ih['+'] * Dm)**5], dim=1),
                    '-': torch.cat([torch.ones((self.n, 1), dtype=self.dtype, device=self.device), (Ih['-'] * Dm), (Ih['-'] * Dm)**2, (Ih['-'] * Dm)**3, (Ih['-'] * Dm)**4, (Ih['-'] * Dm)**5], dim=1)}
        self.Γ_5 = {'+': (1 / self.n) * (self.R_5['+'].T @ self.K['+'] @ self.R_5['+']), '-': (1 / self.n) * (self.R_5['-'].T @ self.K['-'] @ self.R_5['-'])}
        self.Γ_5_inv = {'+': torch.linalg.pinv(self.Γ_5['+']), '-': torch.linalg.pinv(self.Γ_5['-'])}

        self.B_2β = {'+': self.Γ_2_inv['+'] @ self.R_2['+'].T @ self.L['+'] / self.n @ self.Y, 
                  '-': self.Γ_2_inv['-'] @ self.R_2['-'].T @ self.L['-'] / self.n @ self.Y}
        self.H_1β = {'+': self.Γ_1_inv['+'] @ self.R_1['+'].T @ self.K['+'] / self.n @ self.Y, 
                  '-': self.Γ_1_inv['-'] @ self.R_1['-'].T @ self.K['-'] / self.n @ self.Y}
        self.ε = {'+': (self.Y - self.R_1['+'] @ self.H_1β['+']),  # (n, 1)
                  '-': (self.Y - self.R_1['-'] @ self.H_1β['-'])}  # (n, 1)
        self.σ = {'+': (self.Y - self.R_2['+'] @ self.B_2β['+']).abs(),  # (n, 1)
                  '-': (self.Y - self.R_2['+'] @ self.B_2β['+']).abs(),}  # (n, 1)
        self.Σ = {'+': torch.diag(self.σ['+'].flatten()**2),
                  '-': torch.diag(self.σ['-'].flatten()**2)}
        
        self.P_bc = {'+': self.Γ_1_inv['+'] @ self.R_1['+'].T @ self.K['+'] - (self.h['+'] / self.b['+'])**2 * self.Γ_1_inv['+'] @ self.Λ_1['+'] @ self.e_2.T @ self.Γ_2_inv['+'] @ self.R_2['+'].T @ self.L['+'], 
                     '-': self.Γ_1_inv['-'] @ self.R_1['-'].T @ self.K['-'] - (self.h['-'] / self.b['-'])**2 * self.Γ_1_inv['-'] @ self.Λ_1['-'] @ self.e_2.T @ self.Γ_2_inv['-'] @ self.R_2['-'].T @ self.L['-']}
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
        term5 = self.v_rbc[sn]**(-4) * mean5a @ mean5b * (z * (z**2 - 1))
        
        mean6 = torch.mean(self.ℓ_0_bc[sn]**2 * (self.𝛿[sn] * self.R_2[sn] @ self.Γ_2_inv[sn] * self.R_2[sn]).sum(dim = 1, keepdim = True)**2 * self.ε[sn]**2)
        term6 = self.v_rbc[sn]**(-2) * mean6 * (z * (z**2 - 1)/4)
        
        mean7a = torch.mean(self.ℓ_0_bc[sn] * self.ε[sn]**2 * self.𝛿[sn] * (self.R_2[sn] @ self.Γ_2_inv[sn]), dim = 0, keepdim = True)
        mean7b = torch.mean((self.ℓ_0_bc[sn]**2).view(self.n, 1, 1) * torch.bmm(self.R_2[sn].unsqueeze(2), (self.R_2[sn] @ self.Γ_2_inv[sn]).unsqueeze(2).mT), dim = 0) / self.b[sn]
        mean7c = torch.mean(self.𝛿[sn] * self.R_2[sn] * self.ℓ_0_bc[sn] * self.ε[sn]**2, dim = 0, keepdim = True).T
        term7 = self.v_rbc[sn]**(-4) * (mean7a @ mean7b @ mean7c) * (z * (z**2 - 1)/2)
        
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
        𝛼_3 = (1/self.n) * self.e_3.T @ self.Γ_5_inv[sn] @ self.R_5[sn].T @ self.K[sn] @ self.Y
        return 𝛼_3
    
    def __get_bandwidth(self, optim_mode = 'newton-cg', tol = 0.001):
        def obj(h):
            self.h['-'] = h[0]
            self.h['+'] = h[1]
            if torch.min(h) <= 0 or torch.max(h) > 10 * torch.max(torch.abs(self.D - self.cutoff)):
                return torch.inf
            self.b = {'+': 1/self.ρ * self.h['+'], '-': 1/self.ρ * self.h['-']}
            self.__build_matrices()
            self.__build_edgeworth_terms()
            𝜇_3 = {'+': self.__get_𝜇_3('+'), '-': self.__get_𝜇_3('-')}
            𝜂_bc = {'+': torch.sqrt(self.n * self.h['+']) * self.h['+']**3 * 𝜇_3['+'] / factorial(3) * self.e_0.T @ self.Γ_1_inv['+'] @ (self.Λ_1_2['+'] - self.Λ_1['+'] @ self.e_2.T @ self.Γ_2_inv['+'] @ self.Λ_2_1['+']),
                    '-': torch.sqrt(self.n * self.h['-']) * self.h['-']**3 * 𝜇_3['-'] / factorial(3) * self.e_0.T @ self.Γ_1_inv['-'] @ (self.Λ_1_2['-'] - self.Λ_1['-'] @ self.e_2.T @ self.Γ_2_inv['-'] @ self.Λ_2_1['-'])}
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
        resid_pos = self.Y - self.R_2['+'] @ self.Γ_2_inv['+'] @ self.R_2['+'].T @ self.L['+'] / self.n @ self.Y
        resid_neg = self.Y - self.R_2['-'] @ self.Γ_2_inv['-'] @ self.R_2['-'].T @ self.L['-'] / self.n @ self.Y
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
    