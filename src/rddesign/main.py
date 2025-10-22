import numpy as np, pandas as pd, warnings, torch
from scipy.stats import norm
from helpers import *
from itertools import permutations

def triangular_kernel(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float32)
    a = np.abs(u)
    w = np.where(a <= 1.0, 1.0 - a, 0.0).astype(np.float32)
    return w

class pdd:
    """
    
    Notes
    -----
    - 
    """
    def __init__(self, Y, W, D, Z, cutoff=0.0, alpha=0.05, kernel='triangle', dtype=np.float32):
        self.dtype = dtype
        self.Y = np.asarray(Y, dtype=dtype).reshape(-1,1)
        if self.Y.ndim == 1: self.Y = self.Y.reshape(-1,1)
        self.W = np.asarray(W, dtype=dtype)
        if self.W.ndim == 1: self.W = self.W.reshape(-1,1)
        self.D = np.asarray(D, dtype=dtype)
        if self.D.ndim == 1: self.D = self.D.reshape(-1,1)
        self.Z = np.asarray(Z, dtype=dtype)
        if self.Z.ndim == 1: self.Z = self.Z.reshape(-1,1)
        self.n = int(self.D.shape[0])
        self.cutoff = dtype(cutoff)
        if kernel == 'triangle':
            self.kernel = triangular_kernel
        self.alpha = dtype(alpha)
        # stacked S=(Y,W1,...,Wq)
        self.S = np.concatenate([self.Y, self.W], axis=1).astype(dtype)  # n x (q+1)
        self.vecS = self.S.flatten().reshape(-1,1)
        self.q = int(self.S.shape[1]) - 1
        # selection s-hat gets built after gamma_hat_-
        self.s_hat_vec = None
        # bandwidths
        self.h = None
        self.b = None

    def __idx(self, i, j):
        if j < i:
            return i*(self.n-1) + j
        else:  # j >= i
            return i*(self.n-1) + (j-1)

    def _build_matrices(self):
        self.R_1 = np.concatenate([np.ones((self.n, 1)), (self.D - self.cutoff)/self.h], axis=1).astype(self.dtype)     # R^{n x 2}
        self.R_2 = np.concatenate([np.ones((self.n, 1)), (self.D - self.cutoff)/self.b, ((self.D - self.cutoff)/self.b)**2], axis=1).astype(self.dtype) # R^{n x 2}
        self.𝜔 = {'+': ((1/self.h) * np.where(self.D >= self.cutoff, 1, 0) * self.kernel((self.D - self.cutoff)/self.h)).astype(self.dtype),
                  '-': ((1/self.h) * np.where(self.D < self.cutoff, 1, 0) * self.kernel((self.D - self.cutoff)/self.h)).astype(self.dtype) } # R^{n x 1}
        self.𝛿 = {'+': ((1/self.b) * np.where(self.D >= self.cutoff, 1, 0) * self.kernel((self.D - self.cutoff)/self.b)).astype(self.dtype),
                  '-': ((1/self.b) * np.where(self.D < self.cutoff, 1, 0) * self.kernel((self.D - self.cutoff)/self.b)).astype(self.dtype)}  # R^{n x 1}
        self.K = {'+': np.diag(self.𝜔['+'].flatten()), '-': np.diag(self.𝜔['-'].flatten())}   # R^{n x n}
        self.L = {'+': np.diag(self.𝛿['+'].flatten()), '-': np.diag(self.𝛿['-'].flatten())}   # R^{n x n}
        self.Q = self.Z.T @ self.K['-'] @ (np.eye(self.n).astype(self.dtype) - self.R_1 @ np.linalg.pinv(self.R_1.T @ self.K['-'] @ self.R_1) @ self.R_1 @ self.K['-'] @ self.W)   # R^{q x q}
        self.Γ_1 = {'+': (1/self.n) * self.R_1.T @ self.K['+'] @ self.R_1, '-': (1/self.n) * self.R_1.T @ self.K['-'] @ self.R_1}   # R^{2 x 2}
        self.Γ_2 = {'+': (1/self.n) * self.R_2.T @ self.L['+'] @ self.R_2, '-': (1/self.n) * self.R_2.T @ self.L['-'] @ self.R_2}   # R^{3 x 3}
        self.Γ_1_inv = {'+': np.linalg.pinv((1/self.n) * self.R_1.T @ self.K['+'] @ self.R_1), '-': np.linalg.pinv((1/self.n) * self.R_1.T @ self.K['-'] @ self.R_1)}   # R^{2 x 2}
        self.Γ_2_inv = {'+': np.linalg.pinv((1/self.n) * self.R_2.T @ self.L['+'] @ self.R_2), '-': np.linalg.pinv((1/self.n) * self.R_2.T @ self.L['-'] @ self.R_2)}   # R^{3 x 3}
        self.Λ_1 = {'+': (1/self.n) * self.R_1.T @ self.K['+'] @ ((1/self.h) * (self.D - self.cutoff))**2, '-': (1/self.n) * self.R_1.T @ self.K['-'] @ ((1/self.h) * (self.D - self.cutoff))**2}   # R^{2 x 1}
        self.Λ_2 = {'+': (1/self.n) * self.R_2.T @ self.L['+'] @ ((1/self.b) * (self.D - self.cutoff))**2, '-': (1/self.n) * self.R_2.T @ self.L['-'] @ ((1/self.b) * (self.D - self.cutoff))**2}   # R^{3 x 1}
        self.Λ_1_2 = {'+': (1/self.n) * self.R_1.T @ self.K['+'] @ ((1/self.h) * (self.D - self.cutoff))**3, '-': (1/self.n) * self.R_1.T @ self.K['-'] @ ((1/self.h) * (self.D - self.cutoff))**3}    # R^{2 x 1}
        self.Λ_2_2 = {'+': (1/self.n) * self.R_2.T @ self.L['+'] @ ((1/self.b) * (self.D - self.cutoff))**3, '-': (1/self.n) * self.R_2.T @ self.L['-'] @ ((1/self.b) * (self.D - self.cutoff))**3}    # R^{3 x 1}
        self.e_2 = np.array([[0], [0], [1]])
        self.l_q1 = np.ones((self.q + 1, 1))
        self.e_3 = np.array([[0], [0], [0], [1], [0], [0]])
        self.R_5 = np.concatenate([np.ones((self.n, 1)), (self.D - self.cutoff)/self.b, ((self.D - self.cutoff)/self.b)**2, ((self.D - self.cutoff)/self.b)**3,
                                   ((self.D - self.cutoff)/self.b)**4, ((self.D - self.cutoff)/self.b)**5], axis = 1).astype(self.dtype) # R^{n x 5}
        self.Γ_5 = {'+': (1/self.n) * self.R_5.T @ self.K['+'] @ self.R_5, '-': (1/self.n) * self.R_5.T @ self.K['-'] @ self.R_5}   # R^{5 x 5}
        self.Γ_5_inv = {'+': np.linalg.pinv((1/self.n) * self.R_5.T @ self.K['+'] @ self.R_5), '-': np.linalg.pinv((1/self.n) * self.R_5.T @ self.K['-'] @ self.R_5)}   # R^{5 x 5}
        
        self.𝛾 = (np.linalg.pinv(self.Q) @ self.Z.T @  self.K['-'] @ (np.eye(self.n) - self.R_1 @ np.linalg.pinv(self.R_1.T @ self.K['-'] @ self.R_2) @ self.R_2.T @ self.K['-']) @ self.Y).flatten()
        self.s = np.concatenate([np.array([[1, 0]]), np.array([[𝛾, 0] for 𝛾 in self.𝛾])], axis = 0).flatten().reshape(-1, 1)
        
        vecB2β = {'+': np.kron(np.eye(self.q + 1), self.Γ_2_inv['+'] @ self.R_2.T @ self.L['+'] / self.n) @ self.vecS, 
                  '-': np.kron(np.eye(self.q + 1), self.Γ_2_inv['-'] @ self.R_2.T @ self.L['-'] / self.n) @ self.vecS}
        self.ε = {'+': [self.S[i, None].T - np.kron(np.eye(self.q + 1), self.R_2[i, None]) @ vecB2β['+'] for i in range(self.n)],
                  '-': [self.S[i, None].T - np.kron(np.eye(self.q + 1), self.R_2[i, None]) @ vecB2β['-'] for i in range(self.n)]}
        self.σ = {'+': [np.sqrt(self.ε['+'][i]**2) for i in range(self.n)],
                  '-': [np.sqrt(self.ε['-'][i]**2) for i in range(self.n)]}
        self.Σ = {'+': np.diag(np.array([[self.σ['+'][i][j, 0]**2 for i in range(self.n)] for j in range(self.q + 1)]).flatten()), 
                  '-': np.diag(np.array([[self.σ['-'][i][j, 0]**2 for i in range(self.n)] for j in range(self.q + 1)]).flatten())}
        P_bc = {'+': self.Γ_1_inv['+'] @ self.R_1.T @ self.K['+'] - (self.h/self.b)**2 * self.Γ_1_inv['+'] @ self.Λ_1['+'] @ self.e_2.T @ self.Γ_2_inv['+'] @ self.R_2.T @ self.L['+'],
                '-': self.Γ_1_inv['-'] @ self.R_1.T @ self.K['-'] - (self.h/self.b)**2 * self.Γ_1_inv['-'] @ self.Λ_1['-'] @ self.e_2.T @ self.Γ_2_inv['-'] @ self.R_2.T @ self.L['-']}
        self.v_bc = np.sqrt((self.h/self.n) * self.s.T @ np.kron(np.eye(self.q + 1), P_bc['+']) @ self.Σ['+'] @ np.kron(np.eye(self.q + 1), P_bc['+']) @ self.s + \
                    (self.h/self.n) * self.s.T @ np.kron(np.eye(self.q + 1), P_bc['-']) @ self.Σ['-'] @ np.kron(np.eye(self.q + 1), P_bc['-']) @ self.s)
        
    def _build_edgeworth_terms(self):
        self.ℓ_0_us = {'+': [self.h * self.𝜔['+'][i, 0] * self.Γ_1_inv['+'] @ self.R_1[i, :].T for i in range(self.n)],
                       '-': [self.h * self.𝜔['-'][i, 0] * self.Γ_1_inv['-'] @ self.R_1[i, :].T for i in range(self.n)]} # R^{2 x 1} x n
        self.ℓ_0_bc = {'+': [self.ℓ_0_us['+'][i] - \
            self.b * (self.h/self.b)**2 * self.𝛿['+'][i, 0] * self.Γ_1_inv['+'] @ self.Λ_1['+'] @ self.e_2.T @ self.Γ_2_inv['+'] @ self.R_2[i, :].T for i in range(self.n)],
                         '-': [self.ℓ_0_us['-'][i] - \
            self.b * (self.h/self.b)**2 * self.𝛿['-'][i, 0] * self.Γ_1_inv['-'] @ self.Λ_1['-'] @ self.e_2.T @ self.Γ_2_inv['-'] @ self.R_2[i, :].T for i in range(self.n)]} # R^{2 x 1} x n
        
        self.ℓ_1_us = {'+': [self.h**2 * self.𝜔['+'][i, 0] * \
            self.Γ_1_inv['+'] @ (self.Γ_1['+'] - self.𝜔['+'][j, 0] * self.R_1[j, None].T @ self.R_1[j, None]) @ self.Γ_1_inv['+'] @ self.R_1[i, :].T \
            for i, j in permutations(range(self.n), 2)],
                         '-': [self.h**2 * self.𝜔['-'][i, 0] * \
            self.Γ_1_inv['-'] @ (self.Γ_1['-'] - self.𝜔['-'][j, 0] * self.R_1[j, None].T @ self.R_1[j, None]) @ self.Γ_1_inv['-'] @ self.R_1[i, :].T \
            for i, j in permutations(range(self.n), 2)]} # R^{2 x 1} x n(n - 1)
        
        # Extra term needed for ℓ_1_bc_p
        E𝜔RD = {'+': np.mean([self.𝜔['+'][i, 0] * self.R_1[j, None] * ((self.D[i, 0] - self.cutoff)/self.h)**2 for i, j in permutations(range(self.n), 2)], axis = 0),
                '-': np.mean([self.𝜔['-'][i, 0] * self.R_1[j, None] * ((self.D[i, 0] - self.cutoff)/self.h)**2 for i, j in permutations(range(self.n), 2)], axis = 0)}
        self.ℓ_1_bc = {'+': [self.ℓ_1_us['+'][self.__idx(i, j)] - self.b * (self.h/self.b)**2 * \
            self.Γ_1_inv['+'] @ (self.h * (self.Γ_1['+'] - self.𝜔['+'][j, 0] * self.R_1[j, None].T @ self.R_1[j, None]) @ self.Γ_1_inv['+'] @ self.Λ_1['+'] @ self.e_2.T \
            + self.h * (self.𝜔['+'][i, 0] * self.R_1[j, None] * ((self.D[i, 0] - self.cutoff)/self.h)**2 - E𝜔RD) @ self.e_2.T \
            + self.b * self.Λ_1['+'] @ self.e_2.T @ self.Γ_2_inv['+'] @ (self.Γ_2['+'] - self.𝛿['+'][j, 0] * self.R_2[j, None].T @ self.R_2[j, None]))\
            @ self.Γ_2_inv['+'] @ self.R_2[i, :].T * self.𝛿['+'][i, 0] for i, j in permutations(range(self.n), 2)],
            '-': [self.ℓ_1_us['-'][self.__idx(i, j)] - self.b * (self.h/self.b)**2 * \
            self.Γ_1_inv['-'] @ (self.h * (self.Γ_1['-'] - self.𝜔['-'][j, 0] * self.R_1[j, None].T @ self.R_1[j, None]) @ self.Γ_1_inv['-'] @ self.Λ_1['-'] @ self.e_2.T \
            + self.h * (self.𝜔['-'][i, 0] * self.R_1[j, None] * ((self.D[i, 0] - self.cutoff)/self.h)**2 - E𝜔RD) @ self.e_2.T \
            + self.b * self.Λ_1['-'] @ self.e_2.T @ self.Γ_2_inv['-'] @ (self.Γ_2['-'] - self.𝛿['-'][j, 0] * self.R_2[j, None].T @ self.R_2[j, None]))\
            @ self.Γ_2_inv['-'] @ self.R_2[i, :].T * self.𝛿['-'][i, 0] for i, j in permutations(range(self.n), 2)]}
        
    def __get_q_1(self, sn, α = 0.05):
        z = norm.ppf(1 - α/2)
        term1 = self.v_bc**(-6) * np.mean([(1/self.h) * (self.s.T @ np.kron(self.ε[sn][i], self.ℓ_0_bc[sn][i]))**3 for i in range(self.n)])**2 *\
                (z**3/3 + 7*z/4 + self.v_bc**2 * z * (z**2 - 3)/4)
        term2 = self.v_bc**(-2) * np.mean([(1/self.h) * (self.s.T @ np.kron(self.ε[sn][i], self.ℓ_0_bc[sn][i])) * (self.s.T @ np.kron(self.ε[sn][i], self.ℓ_1_bc[sn][self.__idx(i, i)])) for i in range(self.n)]) *\
                (-(z**2 - 3)/4)
        term3 = self.v_bc**(-4) * np.mean([(1/self.h) * ((self.s.T @ np.kron(self.ε[sn][i], self.ℓ_0_bc[sn][i]))**4 - (self.s.T @ np.kron(self.σ[sn][i], self.ℓ_0_bc[sn][i]))**4 ) for i in range(self.n)]) *\
                (z * (z**2 - 3)/8)
        term4 = self.v_bc**(-2) * np.mean([(1/self.h) * (self.s.T @ np.kron(self.ε[sn][i], self.ℓ_0_bc[sn][i]))**2 * self.b *  self.𝛿[sn][i, 0] * self.R_2[i, None] @ self.Γ_2_inv[sn] @ self.R_2[i, None].T for i in range(self.n)]) *\
                (z * (z**2 - 1)/2)
        term5 = self.v_bc**(-4) * np.mean([(1/self.h) * (self.s.T @ np.kron(self.ε[sn][i], self.ℓ_0_bc[sn][i]))**2 * (self.s.T @ np.kron(self.l_q1, self.ℓ_0_bc[sn][i])) * self.R_2[i, None] @ self.Γ_2_inv[sn] for i in range(self.n)]) *\
                np.mean([(self.b/self.h) * self.𝛿[sn][i, 0] * self.R_2[i, None].T * (self.s.T @ np.kron(self.ε[sn][i]**2, self.ℓ_0_bc[sn][i])) for i in range(self.n)]) *\
                (z * (z**2 - 1))
        term6 = self.v_bc**(-2) * np.mean([(1/self.h)**2 * (self.s.T @ np.kron(self.ε[sn][i], self.ℓ_0_bc[sn][i]))**2 * (self.b * self.𝛿[sn][i, 0] * self.R_2[i, None] @ self.Γ_2_inv[sn] @ self.R_2[i, None].T)**2 for i in range(self.n)]) *\
                (z (z**2 - 1))/4
        term7 = self.v_bc**(-4) * np.mean([(1/self.h) * (self.s.T @ np.kron(self.ε[sn][i]**2, self.ℓ_0_bc[sn][i])) * self.b * self.𝛿[sn][i, 0] * self.R_2[i, None] @ self.Γ_2_inv[sn] for i in range(self.n)]) *\
                np.mean([(1/self.h) * (self.s.T @ np.kron(self.l_q1, self.ℓ_0_bc[sn][i]))**2 * self.R_2[i, None].T @ self.R_2[i, None] @ self.Γ_2_inv[sn] for i in range(self.n)]) *\
                    np.mean([(self.b/self.h) * self.𝛿[sn][i, 0] * self.R_2[i, None].T * (self.s.T @ np.kron(self.ε[sn][i]**2, self.ℓ_0_bc[sn][i])) for i in range(self.n)]) *\
                    (z * (z**2 - 1)/2)
        term8 = self.v_bc**(-4) * np.mean([ (1/self.h) * (self.s.T @ np.kron(self.ε[sn][i], self.ℓ_0_bc[sn][i]))**4 for i in range(self.n)]) *\
                (-z * (z**2 - 3)/24)
        diff = np.mean([(self.s.T @ np.kron(self.σ[sn][i], self.ℓ_0_bc[sn][i]))**2 for i in range(self.n)])
        term9 = self.v_bc**(-4) * np.mean([(1/self.h) * ((self.s.T @ np.kron(self.σ[sn][i], self.ℓ_0_bc[sn][i]))**2 - diff) * (self.s.T @ np.kron(self.ε[sn][i], self.ℓ_0_bc[sn][i]))**2 for i in range(self.n)]) *\
                (z * (z**2 - 1)/4)
        term10 = self.v_bc**(-4) * np.mean([(1/self.h)**2 * (self.s.T @ np.kron(self.σ[sn][i], self.ℓ_1_bc[sn][self.__idx(i, j)])) * (self.s.T @ np.kron(self.σ[sn][i], self.ℓ_0_bc[sn][i])) * (self.s.T @ np.kron(self.ε[sn][i], self.ℓ_0_bc[sn][i]))**2 \
                for i, j in permutations(range(self.n), 2)]) * (z * (z**2 - 3))
        term11 = self.v_bc**(-4) * np.mean([(1/self.h)**2 * (self.s.T @ np.kron(self.ε[sn][i], self.ℓ_0_bc[sn][i])) * (self.s.T @ np.kron(self.ε[sn][i], self.ℓ_1_bc[sn][self.__idx(i, j)])) * ((self.s.T @ np.kron(self.σ[sn][i], self.ℓ_0_bc[sn][i]))**2 - diff) \
                for i, j in permutations(range(self.n), 2)]) * (z * (z**2 - 3))
        term12 = self.v_bc**(-4) * np.mean([(1/self.h) * ((self.s.T @ np.kron(self.σ[sn][i], self.ℓ_0_bc[sn][i]))**2 - diff) for i in range(self.n)]) *\
                (-z * (z**2 + 1)/8)
        q_1 = term1 + term2 + term3 - term4 - term5 + term6 + term7 + term8 + term9 + term10 + term11 + term12
        return q_1
        
            
    def __fit_bandwidth(self):
        pass