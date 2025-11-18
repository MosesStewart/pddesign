import numpy as np, pandas as pd, warnings, torch
from scipy.stats import norm
from helpers import *
from itertools import permutations
from torchmin import minimize
from math import factorial

def triangular_kernel(u: torch.Tensor) -> torch.Tensor:
    u = torch.as_tensor(u, dtype=torch.float32)
    a = torch.abs(u)
    return torch.where(a <= 1.0, 1.0 - a, torch.zeros_like(a)).to(torch.float32)


class pdd:
    def __init__(self, Y, W, D, Z, cutoff=0.0, alpha=0.05, kernel='triangle', dtype=torch.float32):
        self.dtype = dtype
        self.Y = torch.as_tensor(Y, dtype=dtype)
        if self.Y.ndim == 1: self.Y = self.Y.reshape(-1, 1)
        self.W = torch.as_tensor(W, dtype=dtype)
        if self.W.ndim == 1: self.W = self.W.reshape(-1, 1)
        self.D = torch.as_tensor(D, dtype=dtype)
        if self.D.ndim == 1: self.D = self.D.reshape(-1, 1)
        self.Z = torch.as_tensor(Z, dtype=dtype)
        if self.Z.ndim == 1: self.Z = self.Z.reshape(-1, 1)
        self.n = int(self.D.shape[0])
        self.cutoff = torch.tensor(cutoff, dtype=dtype)
        self.alpha = torch.tensor(alpha, dtype=dtype)
        if kernel == 'triangle': self.kernel = triangular_kernel
        self.S = torch.cat([self.Y, self.W], dim=1).to(dtype)
        self.vecS = self.S.flatten().reshape(-1, 1)
        self.vecW = self.W.flatten().reshape(-1, 1)
        self.q = int(self.S.shape[1]) - 1
        self.s_hat_vec = None
        self.h = {'+': torch.std(self.D) * self.n**(-1/5), '-': torch.std(self.D) * self.n**(-1/5)}
        self.b = self.h

    def __idx(self, i, j):
        if j < i:
            return i * (self.n - 1) + j
        else:
            return i * (self.n - 1) + (j - 1)

    def __build_matrices(self):
        one_n = torch.ones((self.n, 1), dtype=self.dtype)
        Ih, Ib, Dm = {'+': 1 / self.h['+'], '-': 1 / self.h['-']}, {'+': 1 / self.b['+'], '-': 1 / self.b['-']}, self.D - self.cutoff
        self.R_1 = {'+': torch.cat([one_n, Ih['+'] * Dm], dim=1).to(self.dtype), '-': torch.cat([one_n, Ih['-'] * Dm], dim=1).to(self.dtype)}
        self.R_2 = {'+': torch.cat([one_n, Ib['+'] * Dm, (Ib['+'] * Dm)**2], dim=1).to(self.dtype), '-': torch.cat([one_n, Ib['-'] * Dm, (Ib['-'] * Dm)**2], dim=1).to(self.dtype)}

        self.ind = {'+': (self.D >= self.cutoff).to(self.dtype), '-': (self.D < self.cutoff).to(self.dtype)}
        self.𝜔 = {'+': (Ih['+'] * self.ind['+'] * self.kernel(Ih['+'] * Dm)).to(self.dtype), '-': (Ih['-'] * self.ind['-'] * self.kernel(Ih['-'] * Dm)).to(self.dtype)}
        self.𝛿 = {'+': (Ib * self.ind['+'] * self.kernel(Ib['+'] * Dm)).to(self.dtype), '-': (Ib * self.ind['-'] * self.kernel(Ib['-'] * Dm)).to(self.dtype)}
        self.K = {'+': torch.diag(self.𝜔['+'].flatten()), '-': torch.diag(self.𝜔['-'].flatten())}
        self.L = {'+': torch.diag(self.𝛿['+'].flatten()), '-': torch.diag(self.𝛿['-'].flatten())}
        
        I_n = torch.eye(self.n, dtype=self.dtype)
        self.Q = self.Z.T @ self.K['-'] @ (I_n - self.R_1['-'] @ torch.linalg.pinv(self.R_1['-'].T @ self.K['-'] @ self.R_1['-']) @ self.R_1['-'].T) @ self.K['-'] @ self.W
        self.Γ_1 = {'+': (1 / self.n) * (self.R_1['+'].T @ self.K['+'] @ self.R_1['+']), '-': (1 / self.n) * (self.R_1['-'].T @ self.K['-'] @ self.R_1['-'])}
        self.Γ_2 = {'+': (1 / self.n) * (self.R_2['+'].T @ self.L['+'] @ self.R_2['+']), '-': (1 / self.n) * (self.R_2['-'].T @ self.L['-'] @ self.R_2['-'])}
        self.Γ_1_inv = {'+': torch.linalg.pinv(self.Γ_1['+']), '-': torch.linalg.pinv(self.Γ_1['-'])}
        self.Γ_2_inv = {'+': torch.linalg.pinv(self.Γ_2['+']), '-': torch.linalg.pinv(self.Γ_2['-'])}
        self.Λ_1 = {'+': (1 / self.n) * (self.R_1['+'].T @ self.K['+'] @ (Ih['+'] * Dm)**2), '-': (1 / self.n) * (self.R_1['-'].T @ self.K['-'] @ (Ih['-'] * Dm)**2)}
        self.Λ_2 = {'+': (1 / self.n) * (self.R_2['+'].T @ self.L['+'] @ (Ib['+'] * Dm)**2), '-': (1 / self.n) * (self.R_2['-'].T @ self.L['-'] @ (Ib['-'] * Dm)**2)}
        self.Λ_1_2 = {'+': (1 / self.n) * (self.R_1['+'].T @ self.K['+'] @ (Ih['+'] * Dm)**3), '-': (1 / self.n) * (self.R_1['-'].T @ self.K['-'] @ (Ih['-'] * Dm)**3)}
        self.Λ_2_2 = {'+': (1 / self.n) * (self.R_2['+'].T @ self.L['+'] @ (Ib['+'] * Dm)**3), '-': (1 / self.n) * (self.R_2['-'].T @ self.L['-'] @ (Ib['-'] * Dm)**3)}
        self.e_2 = torch.tensor([[0.0], [0.0], [1.0]], dtype=self.dtype)
        self.l_q1 = torch.ones((self.q + 1, 1), dtype=self.dtype)
        self.e_3 = torch.tensor([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0]], dtype=self.dtype)

        self.R_5 = {'+': torch.cat([torch.ones((self.n, 1), dtype=self.dtype), (Ih['+'] * Dm), (Ih['+'] * Dm)**2, (Ih['+'] * Dm)**3, (Ih['+'] * Dm)**4, (Ih['+'] * Dm)**5], dim=1),
                    '-': torch.cat([torch.ones((self.n, 1), dtype=self.dtype), (Ih['-'] * Dm), (Ih['-'] * Dm)**2, (Ih['-'] * Dm)**3, (Ih['-'] * Dm)**4, (Ih['-'] * Dm)**5], dim=1)}
        self.Γ_5 = {'+': (1 / self.n) * (self.R_5['+'].T @ self.K['+'] @ self.R_5['+']), '-': (1 / self.n) * (self.R_5['-'].T @ self.K['-'] @ self.R_5['-'])}
        self.Γ_5_inv = {'+': torch.linalg.pinv(self.Γ_5['+']), '-': torch.linalg.pinv(self.Γ_5['-'])}
        self.𝛾 = (torch.linalg.pinv(self.Q) @ self.Z.T @ self.K['-'] @ (torch.eye(self.n, dtype=self.dtype) - self.R_1['-'] @ torch.linalg.pinv(self.R_1['-'].T @ self.K['-'] @ self.R_2['-']) @ self.R_2['-'].T @ self.K['-']) @ self.Y).flatten()
        self.s = torch.cat([torch.tensor([[1.0, 0.0]], dtype=self.dtype), torch.stack([torch.tensor([𝛾.item(), 0.0], dtype=self.dtype) for 𝛾 in self.𝛾])], dim=0).flatten().reshape(-1, 1)

        vecB2β = {'+': torch.kron(torch.eye(self.q + 1, dtype=self.dtype), self.Γ_2_inv['+'] @ self.R_2['+'].T @ self.L['+'] / self.n) @ self.vecS, 
                  '-': torch.kron(torch.eye(self.q + 1, dtype=self.dtype), self.Γ_2_inv['-'] @ self.R_2['-'].T @ self.L['-'] / self.n) @ self.vecS}
        self.ε = {'+': [self.S[i, None].T - torch.kron(torch.eye(self.q + 1, dtype=self.dtype), self.R_2['+'][i, None]) @ vecB2β['+'] for i in range(self.n)], 
                  '-': [self.S[i, None].T - torch.kron(torch.eye(self.q + 1, dtype=self.dtype), self.R_2['-'][i, None]) @ vecB2β['-'] for i in range(self.n)]}
        self.σ = {'+': [torch.sqrt(self.ε['+'][i]**2) for i in range(self.n)], 
                  '-': [torch.sqrt(self.ε['-'][i]**2) for i in range(self.n)]}
        self.Σ = {'+': torch.diag(torch.stack([torch.stack([self.σ['+'][i][j, 0]**2 for i in range(self.n)]) for j in range(self.q + 1)]).flatten()), 
                  '-': torch.diag(torch.stack([torch.stack([self.σ['-'][i][j, 0]**2 for i in range(self.n)]) for j in range(self.q + 1)]).flatten())}
        self.P_bc = {'+': self.Γ_1_inv['+'] @ self.R_1['+'].T @ self.K['+'] - (self.h / self.b)**2 * self.Γ_1_inv['+'] @ self.Λ_1['+'] @ self.e_2.T @ self.Γ_2_inv['+'] @ self.R_2['+'].T @ self.L['+'], 
                '-': self.Γ_1_inv['-'] @ self.R_1['-'].T @ self.K['-'] - (self.h / self.b)**2 * self.Γ_1_inv['-'] @ self.Λ_1['-'] @ self.e_2.T @ self.Γ_2_inv['-'] @ self.R_2['-'].T @ self.L['-']}

        self.v_bc = torch.sqrt((self.h / self.n) * (self.s.T @ torch.kron(torch.eye(self.q + 1, dtype=self.dtype), self.P_bc['+']) @ self.Σ['+'] @ torch.kron(torch.eye(self.q + 1, dtype=self.dtype), self.P_bc['+']) @ self.s) +\
                                (self.h / self.n) * (self.s.T @ torch.kron(torch.eye(self.q + 1, dtype=self.dtype), self.P_bc['-']) @ self.Σ['-'] @ torch.kron(torch.eye(self.q + 1, dtype=self.dtype), self.P_bc['-']) @ self.s))

    def __build_edgeworth_terms(self):
        self.ℓ_0_us = {'+': [self.h * self.𝜔['+'][i, 0] * self.Γ_1_inv['+'] @ self.R_1['+'][i, None].T for i in range(self.n)], 
                       '-': [self.h * self.𝜔['-'][i, 0] * self.Γ_1_inv['-'] @ self.R_1['-'][i, None].T for i in range(self.n)]}
        self.ℓ_0_bc = {'+': [self.ℓ_0_us['+'][i] - self.b * (self.h / self.b)**2 * self.𝛿['+'][i, 0] * self.Γ_1_inv['+'] @ self.Λ_1['+'] @ self.e_2.T @ self.Γ_2_inv['+'] @ self.R_2['+'][i, None].T for i in range(self.n)], 
                       '-': [self.ℓ_0_us['-'][i] - self.b * (self.h / self.b)**2 * self.𝛿['-'][i, 0] * self.Γ_1_inv['-'] @ self.Λ_1['-'] @ self.e_2.T @ self.Γ_2_inv['-'] @ self.R_2['-'][i, None].T for i in range(self.n)]}

        self.ℓ_1_us = {'+': [self.h**2 * self.𝜔['+'][i, 0] * self.Γ_1_inv['+'] @ (self.Γ_1['+'] - self.𝜔['+'][j, 0] * self.R_1['+'][j, None].T @ self.R_1['+'][j, None]) @ self.Γ_1_inv['+'] @ self.R_1['+'][i, None].T for i, j in permutations(range(self.n), 2)], 
                       '-': [self.h**2 * self.𝜔['-'][i, 0] * self.Γ_1_inv['-'] @ (self.Γ_1['-'] - self.𝜔['-'][j, 0] * self.R_1['-'][j, None].T @ self.R_1['-'][j, None]) @ self.Γ_1_inv['-'] @ self.R_1['-'][i, None].T for i, j in permutations(range(self.n), 2)]}
        E𝜔RD = {'+': torch.mean(torch.stack([self.𝜔['+'][i, 0] * self.R_1['+'][j, None] * ((self.D[i, 0] - self.cutoff) / self.h)**2 for i, j in permutations(range(self.n), 2)]), dim=0), 
                '-': torch.mean(torch.stack([self.𝜔['-'][i, 0] * self.R_1['-'][j, None] * ((self.D[i, 0] - self.cutoff) / self.h)**2 for i, j in permutations(range(self.n), 2)]), dim=0)}
        self.ℓ_1_bc = {'+': [self.ℓ_1_us['+'][self.__idx(i, j)] - self.b * (self.h / self.b)**2 * self.Γ_1_inv['+'] @ (self.h * (self.Γ_1['+'] - self.𝜔['+'][j, 0] * self.R_1['+'][j, None].T @ self.R_1['+'][j, None]) @ self.Γ_1_inv['+'] @ self.Λ_1['+'] @ self.e_2.T \
                    + self.h * (self.𝜔['+'][i, 0] * self.R_1['+'][j, None] * ((self.D[i, 0] - self.cutoff) / self.h)**2 - E𝜔RD['+']) @ self.e_2.T \
                    + self.b * self.Λ_1['+'] @ self.e_2.T @ self.Γ_2_inv['+'] @ (self.Γ_2['+'] - self.𝛿['+'][j, 0] * self.R_2['+'][j, None].T @ self.R_2['+'][j, None])) @ self.Γ_2_inv['+'] @ self.R_2['+'][i, None].T * self.𝛿['+'][i, 0] for i, j in permutations(range(self.n), 2)], 
                        '-': [self.ℓ_1_us['-'][self.__idx(i, j)] - self.b * (self.h / self.b)**2 * self.Γ_1_inv['-'] @ (self.h * (self.Γ_1['-'] - self.𝜔['-'][j, 0] * self.R_1['-'][j, None].T @ self.R_1['-'][j, None]) @ self.Γ_1_inv['-'] @ self.Λ_1['-'] @ self.e_2.T \
                    + self.h * (self.𝜔['-'][i, 0] * self.R_1['-'][j, None] * ((self.D[i, 0] - self.cutoff) / self.h)**2 - E𝜔RD['-']) @ self.e_2.T \
                    + self.b * self.Λ_1['-'] @ self.e_2.T @ self.Γ_2_inv['-'] @ (self.Γ_2['-'] - self.𝛿['-'][j, 0] * self.R_2['-'][j, None].T @ self.R_2['-'][j, None])) @ self.Γ_2_inv['-'] @ self.R_2['-'][i, None].T * self.𝛿['-'][i, 0] for i, j in permutations(range(self.n), 2)]}
        
    def __get_q_1(self, sn, α=0.05):
        z = torch.tensor(norm.ppf(1 - α / 2), dtype=self.dtype)
        term1 = self.v_bc**(-6) * torch.mean(torch.stack([(1 / self.h) * (self.s.T @ torch.kron(self.ε[sn][i], self.ℓ_0_bc[sn][i]))**3 for i in range(self.n)]))**2 \
                * (z**3 / 3 + 7 * z / 4 + self.v_bc**2 * z * (z**2 - 3) / 4)
        term2 = self.v_bc**(-2) * torch.mean(torch.stack([(1 / self.h) * (self.s.T @ torch.kron(self.ε[sn][i], self.ℓ_0_bc[sn][i])) * (self.s.T @ torch.kron(self.ε[sn][i], self.ℓ_1_bc[sn][self.__idx(i, i)])) for i in range(self.n)])) \
                * (-(z**2 - 3) / 4)
        term3 = self.v_bc**(-4) * torch.mean(torch.stack([(1 / self.h) * ((self.s.T @ torch.kron(self.ε[sn][i], self.ℓ_0_bc[sn][i]))**4 - (self.s.T @ torch.kron(self.σ[sn][i], self.ℓ_0_bc[sn][i]))**4) for i in range(self.n)])) \
                * (z * (z**2 - 3) / 8)
        term4 = self.v_bc**(-2) * torch.mean(torch.stack([(1 / self.h) * (self.s.T @ torch.kron(self.ε[sn][i], self.ℓ_0_bc[sn][i]))**2 * self.b * self.𝛿[sn][i, 0] * (self.R_2[sn][i, None] @ self.Γ_2_inv[sn] @ self.R_2[sn][i, None].T) for i in range(self.n)])) \
                * (z * (z**2 - 1) / 2)
        term5 = self.v_bc**(-4) * torch.mean(torch.stack([(1 / self.h) * (self.s.T @ torch.kron(self.ε[sn][i], self.ℓ_0_bc[sn][i]))**2 * (self.s.T @ torch.kron(self.l_q1, self.ℓ_0_bc[sn][i])) * (self.R_2[sn][i, None] @ self.Γ_2_inv[sn]) for i in range(self.n)])) \
                * torch.mean(torch.stack([(self.b / self.h) * self.𝛿[sn][i, 0] * self.R_2[sn][i, None].T * (self.s.T @ torch.kron(self.ε[sn][i]**2, self.ℓ_0_bc[sn][i])) for i in range(self.n)])) * (z * (z**2 - 1))
        term6 = self.v_bc**(-2) * torch.mean(torch.stack([((1 / self.h)**2) * (self.s.T @ torch.kron(self.ε[sn][i], self.ℓ_0_bc[sn][i]))**2 * (self.b * self.𝛿[sn][i, 0] * (self.R_2[sn][i, None] @ self.Γ_2_inv[sn] @ self.R_2[sn][i, None].T))**2 for i in range(self.n)])) \
                * (z * (z**2 - 1)) / 4
        term7 = self.v_bc**(-4) * torch.mean(torch.stack([(1 / self.h) * (self.s.T @ torch.kron(self.ε[sn][i]**2, self.ℓ_0_bc[sn][i])) * self.b * self.𝛿[sn][i, 0] * (self.R_2[sn][i, None] @ self.Γ_2_inv[sn]) for i in range(self.n)])) \
                * torch.mean(torch.stack([(1 / self.h) * (self.s.T @ torch.kron(self.l_q1, self.ℓ_0_bc[sn][i]))**2 * (self.R_2[sn][i, None].T @ self.R_2[sn][i, None] @ self.Γ_2_inv[sn]) for i in range(self.n)])) \
                * torch.mean(torch.stack([(self.b / self.h) * self.𝛿[sn][i, 0] * self.R_2[sn][i, None].T * (self.s.T @ torch.kron(self.ε[sn][i]**2, self.ℓ_0_bc[sn][i])) for i in range(self.n)])) * (z * (z**2 - 1) / 2)
        term8 = self.v_bc**(-4) * torch.mean(torch.stack([(1 / self.h) * (self.s.T @ torch.kron(self.ε[sn][i], self.ℓ_0_bc[sn][i]))**4 for i in range(self.n)])) \
                * (-z * (z**2 - 3) / 24)
        diff = torch.mean(torch.stack([(self.s.T @ torch.kron(self.σ[sn][i], self.ℓ_0_bc[sn][i]))**2 for i in range(self.n)]))
        term9 = self.v_bc**(-4) * torch.mean(torch.stack([(1 / self.h) * ((self.s.T @ torch.kron(self.σ[sn][i], self.ℓ_0_bc[sn][i]))**2 - diff) * (self.s.T @ torch.kron(self.ε[sn][i], self.ℓ_0_bc[sn][i]))**2 for i in range(self.n)])) \
                * (z * (z**2 - 1) / 4)
        term10 = self.v_bc**(-4) * torch.mean(torch.stack([((1 / self.h)**2) * (self.s.T @ torch.kron(self.σ[sn][i], self.ℓ_1_bc[sn][self.__idx(i, j)])) * (self.s.T @ torch.kron(self.σ[sn][i], self.ℓ_0_bc[sn][i])) \
                * (self.s.T @ torch.kron(self.ε[sn][i], self.ℓ_0_bc[sn][i]))**2 for i, j in permutations(range(self.n), 2)])) * (z * (z**2 - 3))
        term11 = self.v_bc**(-4) * torch.mean(torch.stack([((1 / self.h)**2) * (self.s.T @ torch.kron(self.ε[sn][i], self.ℓ_0_bc[sn][i])) * (self.s.T @ torch.kron(self.ε[sn][i], self.ℓ_1_bc[sn][self.__idx(i, j)])) \
                * ((self.s.T @ torch.kron(self.σ[sn][i], self.ℓ_0_bc[sn][i]))**2 - diff) for i, j in permutations(range(self.n), 2)])) * (z * (z**2 - 3))
        term12 = self.v_bc**(-4) * torch.mean(torch.stack([(1 / self.h) * ((self.s.T @ torch.kron(self.σ[sn][i], self.ℓ_0_bc[sn][i]))**2 - diff) for i in range(self.n)])) * (-z * (z**2 + 1) / 8)
        q_1 = term1 + term2 + term3 - term4 - term5 + term6 + term7 + term8 + term9 + term10 + term11 + term12
        return q_1

    def __get_q_2(self, sn, α=0.05):
        z = torch.tensor(norm.ppf(1 - α / 2), dtype=self.dtype)
        q_2 = - self.v_bc**(-2) * z / 2
        return q_2
    
    def __get_q_3(self, sn, α=0.05):
        z = torch.tensor(norm.ppf(1 - α / 2), dtype=self.dtype)
        q_3 = - self.v_bc**(-4) * torch.mean(torch.stack([(1 / self.h) * (self.s.T @ torch.kron(self.ε[sn][i], self.ℓ_0_bc[sn][i]))**3 for i in range(self.n)])) * z**3 / 3
        return q_3
    
    def __get_𝜇_3(self, sn):
        𝛼_3 = torch.kron(torch.eye(self.q + 1, dtype=self.dtype), (1/self.n) * self.e_3.T @ self.Γ_5_inv[sn] @ self.R_5 @ self.K[sn]) @ self.vecS # R^{(q + 1)}
        return 𝛼_3
    
    def __get_bandwidth(self, optim_mode = 'newton-exact', tol = 0.001):
        def obj(h):
            self.h['+'] = h[0]
            self.h['-'] = h[1]
            self.__build_matrices()
            self.__build_edgeworth_terms()
            𝜇_3 = {'+': self.__get_𝜇_3('+'), '-': self.__get_𝜇_3('-')}
            𝜂_bc = {'+': torch.sqrt(self.n * self.h['+']) * self.h['+']**3 * self.s.T @ torch.kron(𝜇_3 / factorial(3), self.Γ_1_inv['+'] @ (self.Λ_1_2['+'] - self.Λ_1['+'] @ self.e_2.T @ self.Γ_2_inv['+'] @ self.Λ_2_2['+'])),
                    '-': torch.sqrt(self.n * self.h['-']) * self.h['-']**3 * self.s.T @ torch.kron(𝜇_3 / factorial(3), self.Γ_1_inv['-'] @ (self.Λ_1_2['-'] - self.Λ_1['-'] @ self.e_2.T @ self.Γ_2_inv['-'] @ self.Λ_2_2['-']))}
            q_1 = {'+': self.__get_q_1('+'), '-': self.__get_q_1('-')}
            q_2 = self.__get_q_2()
            q_3 = {'+': self.__get_q_3('+'), '-': self.__get_q_3('-')}
            H = {'+': self.h['+'] * self.n**(1/5), '-': self.h['-'] * self.n**(1/5)}
            loss = ( (1/H['+']) * q_1['+'] + H['+']**9 * 𝜂_bc['+']**2 * q_2 + H['+']**3 * 𝜂_bc['+'] * q_3['+'] )**2 + ((1/H['-']) * q_1['-'] + H['-']**9 * 𝜂_bc['-']**2 * q_2 + H['-']**3 * 𝜂_bc['-'] * q_3['-'])**2
            return loss
        
        h0 = torch.tensor([self.h['+'], self.h['-']])
        res = minimize(obj, h0, method = optim_mode, tol = tol)
        return res
    
    def fit(self):
        res = self.__get_bandwidth()
        P_bc = self.P_bc['+'] - self.P_bc['-']
        est = (1/self.n) * self.s.T @ np.kron(torch.eye(self.q + 1, dtype=self.dtype), P_bc) @ self.vecS
        resid_pos = self.Y - torch.linalg.inv(torch.vstack([self.R_1.T, self.Z.T]).to(self.dtype) @ self.K['+'] @ torch.hstack([self.R_1, self.W]).to(self.dtype)) @ (torch.vstack([self.R_1.T, self.Z.T]).to(self.dtype) @ self.K['+'] @ self.Y)
        resid_neg = self.Y - torch.linalg.inv(torch.vstack([self.R_1.T, self.Z.T]).to(self.dtype) @ self.K['-'] @ torch.hstack([self.R_1, self.W]).to(self.dtype)) @ (torch.vstack([self.R_1.T, self.Z.T]).to(self.dtype) @ self.K['-'] @ self.Y)
        resids = (self.ind['+'] * resid_pos + self.ind['-'] * resid_neg).flatten().cpu().numpy()
        def predict(d):
            d = torch.as_tensor(d, dtype=self.dtype)
            if d.ndim == 1: d = d.reshape(-1, 1)
            Ih, dm, one_m = {'+': 1 / self.h['+'], '-': 1 / self.h['-']}, d - self.cutoff, torch.ones((d.shape[0], 1), dtype=self.dtype)
            sw = self.s[2:, :]
            ind = {'+': (d >= self.cutoff).to(self.dtype), '-': (d < self.cutoff).to(self.dtype)}
            
            r = {'+': torch.cat([one_m, Ih['+'] * dm], dim=1).to(self.dtype),
                 '-': torch.cat([one_m, Ih['-'] * dm], dim=1).to(self.dtype)}
            Yhat = {'+': (1/self.n) * r['+'] @ self.P_bc['+'] @ self.Y,
                    '-': (1/self.n) * r['-'] @ self.P_bc['-'] @ self.Y + (1/self.n) * sw.T @ np.kron(torch.eye(self.q, dtype=self.dtype), P_bc) @ self.vecW}
            pred = ind['+'] * Yhat + ind['-'] * Yhat
            return pred.flatten().cpu().numpy()
            
        res = PDDResults(est = est.item(),
                         se = self.v_bc.item(),
                         resid = resids,
                         bandwidth = {'+': self.h['+'], '-': self.h['-']},
                         predict = predict,
                         status = res.success)
