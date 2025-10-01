import numpy as np, pandas as pd, warnings
from scipy.optimize import minimize, Bounds
from scipy.stats import t, chi2
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
        else:  # j > i
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
        self.Q_m = self.Z.T @ self.K['-'] @ (np.eye(self.n).astype(self.dtype) - self.R_1 @ np.linalg.pinv(self.R_1.T @ self.K['-'] @ self.R_1) @ self.R_1 @ self.K['-'] @ self.W)   # R^{q x q}
        self.Γ_1 = {'+': (1/self.n) * self.R_1.T @ self.K['+'] @ self.R_1, '-': (1/self.n) * self.R_1.T @ self.K['-'] @ self.R_1}   # R^{2 x 2}
        self.Γ_2 = {'+': (1/self.n) * self.R_2.T @ self.L['+'] @ self.R_2, '-': (1/self.n) * self.R_2.T @ self.L['-'] @ self.R_2}   # R^{3 x 3}
        self.Γ_1_inv = {'+': np.linalg.pinv((1/self.n) * self.R_1.T @ self.K['+'] @ self.R_1), '-': np.linalg.pinv((1/self.n) * self.R_1.T @ self.K['-'] @ self.R_1)}   # R^{2 x 2}
        self.Γ_2_inv = {'+': np.linalg.pinv((1/self.n) * self.R_2.T @ self.L['+'] @ self.R_2), '-': np.linalg.pinv((1/self.n) * self.R_2.T @ self.L['-'] @ self.R_2)}   # R^{3 x 3}
        self.Λ_1 = {'+': (1/self.n) * self.R_1.T @ self.K['+'] @ ((1/self.h) * (self.D - self.cutoff))**2, '-': (1/self.n) * self.R_1.T @ self.K['-'] @ ((1/self.h) * (self.D - self.cutoff))**2}   # R^{2 x 1}
        self.Λ_2 = {'+': (1/self.n) * self.R_2.T @ self.L['+'] @ ((1/self.b) * (self.D - self.cutoff))**2, '-': (1/self.n) * self.R_2.T @ self.L['-'] @ ((1/self.b) * (self.D - self.cutoff))**2}   # R^{3 x 1}
        self.Λ_1_2 = {'+': (1/self.n) * self.R_1.T @ self.K['+'] @ ((1/self.h) * (self.D - self.cutoff))**3, '-': (1/self.n) * self.R_1.T @ self.K['-'] @ ((1/self.h) * (self.D - self.cutoff))**3}    # R^{2 x 1}
        self.Λ_2_2 = {'+': (1/self.n) * self.R_2.T @ self.L['+'] @ ((1/self.b) * (self.D - self.cutoff))**3, '-': (1/self.n) * self.R_2.T @ self.L['-'] @ ((1/self.b) * (self.D - self.cutoff))**3}    # R^{3 x 1}
        self.e_2 = np.array([[0], [0], [1]])

    def _build_edgeworth_terms(self):
        self.ℓ_0_us = {'+': [self.h * self.𝜔['+'][i, 0] * self.Γ_1_inv['+'] @ self.R_1[i, :].T for i in range(self.n)],
                       '-': [self.h * self.𝜔['-'][i, 0] * self.Γ_1_inv['-'] @ self.R_1[i, :].T for i in range(self.n)]} # R^{2 x 1} x n
        self.ℓ_0_bc = {'+': [self.ℓ_0_us['+'][i] - \
            self.b * (self.h/self.b)**2 * self.𝛿['+'][i, 0] * self.Γ_1_inv['+'] @ self.Λ_1['+'] @ self.e_2.T @ self.Γ_2_inv['+'] @ self.R_2[i, :].T for i in range(self.n)],
                         '-': [self.ℓ_0_us['-'][i] - \
            self.b * (self.h/self.b)**2 * self.𝛿['-'][i, 0] * self.Γ_1_inv['-'] @ self.Λ_1['-'] @ self.e_2.T @ self.Γ_2_inv['-'] @ self.R_2[i, :].T for i in range(self.n)]} # R^{2 x 1} x n
        
        self.ℓ_1_us = {'+': [self.h**2 * self.𝜔['+'][i, 0] * \
            self.Γ_1_inv['+'] @ (self.Γ_1['+'] - self.𝜔['+'][j, 0] * self.R_1[j, :].T @ self.R_1[j, :]) @ self.Γ_1_inv['+'] @ self.R_1[i, :].T \
            for i, j in permutations(range(self.n), 2)],
                         '-': [self.h**2 * self.𝜔['-'][i, 0] * \
            self.Γ_1_inv['-'] @ (self.Γ_1['-'] - self.𝜔['-'][j, 0] * self.R_1[j, :].T @ self.R_1[j, :]) @ self.Γ_1_inv['-'] @ self.R_1[i, :].T \
            for i, j in permutations(range(self.n), 2)]}# R^{2 x 1} x n(n - 1)
        # Extra term needed for ℓ_1_bc_p
        E𝜔RD = {'+': np.mean([self.𝜔['+'][i, 0] * self.R_1[j, :] * ((self.D[i, 0] - self.cutoff)/self.h)**2 for i, j in permutations(range(self.n), 2)], axis = 0),
                '-': np.mean([self.𝜔['-'][i, 0] * self.R_1[j, :] * ((self.D[i, 0] - self.cutoff)/self.h)**2 for i, j in permutations(range(self.n), 2)], axis = 0)}
        self.ℓ_1_bc = {'+': [self.ℓ_1_us['+'][self.__idx(i, j)] - self.b * (self.h/self.b)**2 * \
            self.Γ_1_inv['+'] @ (self.h * (self.Γ_1['+'] - self.𝜔['+'][j, 0] * self.R_1[j, :].T @ self.R_1[j, :]) @ self.Γ_1_inv['+'] @ self.Λ_1['+'] @ self.e_2.T \
            + self.h * (self.𝜔['+'][i, 0] * self.R_1[j, :] * ((self.D[i, 0] - self.cutoff)/self.h)**2 - E𝜔RD) @ self.e_2.T \
            + self.b * self.Λ_1['+'] @ self.e_2.T @ self.Γ_2_inv['+'] @ (self.Γ_2['+'] - self.𝛿['+'][j, 0] * self.R_2[j, :].T @ self.R_2[j, :]))\
            @ self.Γ_2_inv['+'] @ self.R_2[i, :].T * self.𝛿['+'][i, 0] for i, j in permutations(range(self.n), 2)],
            '-': [self.ℓ_1_us['-'][self.__idx(i, j)] - self.b * (self.h/self.b)**2 * \
            self.Γ_1_inv['-'] @ (self.h * (self.Γ_1['-'] - self.𝜔['-'][j, 0] * self.R_1[j, :].T @ self.R_1[j, :]) @ self.Γ_1_inv['-'] @ self.Λ_1['-'] @ self.e_2.T \
            + self.h * (self.𝜔['-'][i, 0] * self.R_1[j, :] * ((self.D[i, 0] - self.cutoff)/self.h)**2 - E𝜔RD) @ self.e_2.T \
            + self.b * self.Λ_1['-'] @ self.e_2.T @ self.Γ_2_inv['-'] @ (self.Γ_2['-'] - self.𝛿['-'][j, 0] * self.R_2[j, :].T @ self.R_2[j, :]))\
            @ self.Γ_2_inv['-'] @ self.R_2[i, :].T * self.𝛿['-'][i, 0] for i, j in permutations(range(self.n), 2)]}

    def _build_matrices(self):
        pass
    
    def _fit_us(self):
        pass