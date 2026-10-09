import numpy as np, warnings, torch, sys, os, re
from scipy.stats import norm
from scipy.optimize import minimize as scipy_minimize
sys.path.append('/'.join(re.split('/|\\\\', os.path.dirname( __file__ ))[0:-1]))
from rddesign.helpers import *
from math import factorial, log


def local_poly(u: torch.Tensor, w: torch.Tensor, Y: torch.Tensor, p: int, h: torch.Tensor):
    """One-sided local polynomial fit of order p in the scaled running variable u = (D - c)/h with normalized kernel
    weights w (zero off the side and outside the window). Returns the design matrix Γ = (1/n) Σ w R R', the coefficients
    β = Γ⁻¹ (1/n) Σ w R Y (so that β_ν = h^ν μ^(ν)(0)/ν!), the residuals, the sandwich meat Ψ = (h/n) Σ w² R R' e², and
    Λ_{p+1} = (1/n) Σ w R u^{p+1}, the weight vector of the leading bias term."""
    n = u.shape[0]
    R = torch.cat([u**k for k in range(p + 1)], dim=1)
    Γ = (R.T * w.T) @ R / n
    Γ_inv = torch.linalg.pinv(Γ)
    β = Γ_inv @ (R.T * w.T) @ Y / n
    e = (Y - R @ β) * (w > 0)
    Ψ = h * (R.T * (w**2 * e**2).T) @ R / n
    Λ = (R.T * w.T) @ u**(p + 1) / n
    return {'Γ_inv': Γ_inv, 'β': β, 'e': e, 'Ψ': Ψ, 'Λ': Λ}

def hc_weights(leverage: torch.Tensor, k: int, p: int, vce: str) -> torch.Tensor:
    """Multiplier applied to squared residuals e² to estimate σ²_i: 1 (hc0), k/(k - p) (hc1), 1/(1 - h_ii)² (hc3), where h_ii is the
    leverage of observation i in the local fit, k the number of observations with positive weight and p the number of coefficients."""
    if vce == 'hc0':
        return torch.ones_like(leverage)
    if vce == 'hc1':
        return torch.ones_like(leverage) * k / max(k - p, 1)
    if vce == 'hc3':
        return 1 / torch.clamp(1 - leverage, min=1e-3)**2
    raise ValueError("vce must be one of 'hc0', 'hc1', 'hc3'")

def local_fits(D: torch.Tensor, Y: torch.Tensor, cutoff, kernel, h: dict, p: int) -> dict:
    """One-sided local polynomial fits of order p on each side of the cutoff at bandwidths h = {'+', '-'}."""
    out, ind = {}, {'+': (D >= cutoff), '-': (D < cutoff)}
    for sn in ('+', '-'):
        u = (D - cutoff) / h[sn]
        w = torch.nan_to_num(ind[sn] * kernel(u) / h[sn])
        out[sn] = local_poly(u, w, Y, p, h[sn])
    return out

def mse_bandwidth(D: torch.Tensor, Y: torch.Tensor, cutoff, kernel, p: int, ν: int, h_V: dict, h_B: dict, two: bool) -> dict:
    """MSE-optimal bandwidth for the local polynomial estimator of order p of μ^(ν)(0), as in rdbwselect. The design objects Γ, Λ, Ψ
    of that estimator are evaluated at the pilot h_V; the leading bias involves μ^(p+1)(0), estimated by a local polynomial of order
    p + 1 at h_B with variance Var(μ̂^(p+1)) = (p+1)!² e'Γ⁻¹ΨΓ⁻¹e / (n h_B^{3+2p}). With B = μ^(p+1)/(p+1)! ν! e_ν'Γ⁻¹Λ_{p+1},
    V = ν!² e_ν'Γ⁻¹ΨΓ⁻¹e_ν and R = 3 Var(B) (the Imbens–Kalyanaraman regularization),
        h = ((1 + 2ν) V / (2 (p + 1 - ν) (B² + R)))^{1/(2p+3)} n^{-1/(2p+3)};
    with a common bandwidth (two = False), B = B_+ - B_-, V = V_+ + V_-, R = R_+ + R_-."""
    n = D.shape[0]
    fits, fits_B = local_fits(D, Y, cutoff, kernel, h_V, p), local_fits(D, Y, cutoff, kernel, h_B, p + 1)
    B, V, R = {}, {}, {}
    for sn in ('+', '-'):
        e_ν = torch.zeros((p + 1, 1), dtype=D.dtype, device=D.device); e_ν[ν, 0] = 1.0
        deriv = factorial(p + 1) * fits_B[sn]['β'][p + 1, 0] / h_B[sn]**(p + 1)
        var_deriv = factorial(p + 1)**2 * (fits_B[sn]['Γ_inv'][[p + 1], :] @ fits_B[sn]['Ψ'] @ fits_B[sn]['Γ_inv'][[p + 1], :].T)[0, 0] / (n * h_B[sn]**(3 + 2 * p))
        κ = factorial(ν) * (e_ν.T @ fits[sn]['Γ_inv'] @ fits[sn]['Λ'])[0, 0] / factorial(p + 1)
        B[sn], R[sn] = κ * deriv, 3 * κ**2 * var_deriv
        V[sn] = factorial(ν)**2 * (e_ν.T @ fits[sn]['Γ_inv'] @ fits[sn]['Ψ'] @ fits[sn]['Γ_inv'].T @ e_ν)[0, 0]
    rate = n**(-1 / (2 * p + 3))
    if two:
        return {sn: ((1 + 2 * ν) * V[sn] / (2 * (p + 1 - ν) * (B[sn]**2 + R[sn])))**(1 / (2 * p + 3)) * rate for sn in ('+', '-')}
    h = ((1 + 2 * ν) * (V['+'] + V['-']) / (2 * (p + 1 - ν) * ((B['+'] - B['-'])**2 + R['+'] + R['-'])))**(1 / (2 * p + 3)) * rate
    return {'+': h, '-': h}

def curvature_bandwidth(D: torch.Tensor, Y: torch.Tensor, cutoff, kernel, c, two: bool) -> dict:
    """Steps 1–2 of rdbwselect: the MSE-optimal bandwidth b for the local quadratic estimator of μ^(2)(0), using μ^(3)(0) from a local
    cubic fit at its own MSE-optimal bandwidth d, which in turn uses μ^(4)(0) from a quartic fit on the full support."""
    c = {'+': c, '-': c}
    full = {sn: torch.max(torch.abs((D - cutoff)[(D >= cutoff) if sn == '+' else (D < cutoff)])) for sn in ('+', '-')}
    d = mse_bandwidth(D, Y, cutoff, kernel, p=3, ν=3, h_V=c, h_B=full, two=two)
    return mse_bandwidth(D, Y, cutoff, kernel, p=2, ν=2, h_V=c, h_B=d, two=two)

def rot_pilot(D: torch.Tensor, kernel: str) -> torch.Tensor:
    """Rule-of-thumb pilot bandwidth c = C_c min{sd(D), IQR(D)/1.349} n^{-1/5}, as in rdbwselect."""
    C_c = {'triangle': 2.576, 'rectangle': 1.843, 'epanechnikov': 2.34}[kernel]
    n = D.shape[0]
    iqr = torch.quantile(D, 0.75) - torch.quantile(D, 0.25)
    return C_c * torch.minimum(torch.std(D), iqr / 1.349) * n**(-1/5)

class pdd:
    def __init__(self, Y: np.ndarray, W: np.ndarray, D: np.ndarray, Z: np.ndarray, cutoff=0.0, alpha=0.05, kernel='triangle',
                 bandwidth = None, bwselect = 'msetwo', vce = 'hc3', dtype = torch.float64, device = 'cpu', reg = 1e-10):
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
        self.kernel_name = kernel if kernel in ('triangle', 'rectangle') else 'epanechnikov'
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
        # bandwidth selection: 'msetwo' picks (h_+, h_-) minimizing each side's AMSE, 'mse' a common h minimizing the joint AMSE;
        # reg is a floor on the regularized squared bias constant; b = h / ρ throughout
        if bwselect not in ('mse', 'msetwo'):
            raise ValueError("bwselect must be 'mse' or 'msetwo'")
        self.bwselect, self.vce, self.reg = bwselect, vce, reg
        self.h = {'+': 2 * torch.std(self.D), '-': 2 * torch.std(self.D)}
        self.b = {'+': 1/self.ρ * self.h['+'], '-': 1/self.ρ * self.h['-']}

    def __build_matrices(self):
        one_n = torch.ones((self.n, 1), dtype=self.dtype, device=self.device)
        Ih, Ib, Dm = {'+': 1 / self.h['+'], '-': 1 / self.h['-']}, {'+': 1 / self.b['+'], '-': 1 / self.b['-']}, self.D - self.cutoff
        self.ξ = {'+': self.h['+'] / self.b['+'], '-': self.h['-'] / self.b['-']}
        self.R_1 = {'+': torch.cat([one_n, Ih['+'] * Dm], dim=1), '-': torch.cat([one_n, Ih['-'] * Dm], dim=1)}
        self.R_2 = {'+': torch.cat([one_n, Ib['+'] * Dm, (Ib['+'] * Dm)**2], dim=1), '-': torch.cat([one_n, Ib['-'] * Dm, (Ib['-'] * Dm)**2], dim=1)}
        # stacked regressors for the local IV regressions: [R, Z] instruments, [R, W] regressors
        RZ_1 = {'+': torch.cat([self.R_1['+'], self.Z], dim=1), '-': torch.cat([self.R_1['-'], self.Z], dim=1)}
        RW_1 = {'+': torch.cat([self.R_1['+'], self.W], dim=1), '-': torch.cat([self.R_1['-'], self.W], dim=1)}
        RZ_2 = {'+': torch.cat([self.R_2['+'], self.Z], dim=1), '-': torch.cat([self.R_2['-'], self.Z], dim=1)}
        RW_2 = {'+': torch.cat([self.R_2['+'], self.W], dim=1), '-': torch.cat([self.R_2['-'], self.W], dim=1)}

        self.ind = {'+': (self.D >= self.cutoff), '-': (self.D < self.cutoff)}
        # nan_to_num: Inf*0=NaN when 1/h overflows for tiny h; kernel should return 0 there
        self.𝜔 = {'+': torch.nan_to_num(Ih['+'] * self.ind['+'] * self.kernel(Ih['+'] * Dm)),
                  '-': torch.nan_to_num(Ih['-'] * self.ind['-'] * self.kernel(Ih['-'] * Dm))}
        self.𝛿 = {'+': torch.nan_to_num(Ib['+'] * self.ind['+'] * self.kernel(Ib['+'] * Dm)),
                  '-': torch.nan_to_num(Ib['-'] * self.ind['-'] * self.kernel(Ib['-'] * Dm))}

        # design matrices Γ (local linear / quadratic in D) and Ψ (local IV), at bandwidths h and b
        self.Γ_1 = {'+': (1 / self.n) * (self.R_1['+'].T * self.𝜔['+'].T) @ self.R_1['+'], '-': (1 / self.n) * (self.R_1['-'].T * self.𝜔['-'].T) @ self.R_1['-']}
        self.Γ_2 = {'+': (1 / self.n) * (self.R_2['+'].T * self.𝛿['+'].T) @ self.R_2['+'], '-': (1 / self.n) * (self.R_2['-'].T * self.𝛿['-'].T) @ self.R_2['-']}
        self.Ψ_1 = {'+': (1 / self.n) * (RZ_1['+'].T * self.𝜔['+'].T) @ RW_1['+'], '-': (1 / self.n) * (RZ_1['-'].T * self.𝜔['-'].T) @ RW_1['-']}
        self.Ψ_2 = {'+': (1 / self.n) * (RZ_2['+'].T * self.𝛿['+'].T) @ RW_2['+'], '-': (1 / self.n) * (RZ_2['-'].T * self.𝛿['-'].T) @ RW_2['-']}
        self.Γ_1_inv = {'+': torch.linalg.pinv(self.Γ_1['+']), '-': torch.linalg.pinv(self.Γ_1['-'])}
        self.Γ_2_inv = {'+': torch.linalg.pinv(self.Γ_2['+']), '-': torch.linalg.pinv(self.Γ_2['-'])}
        self.Ψ_1_inv = {'+': torch.linalg.pinv(self.Ψ_1['+']), '-': torch.linalg.pinv(self.Ψ_1['-'])}
        self.Ψ_2_inv = {'+': torch.linalg.pinv(self.Ψ_2['+']), '-': torch.linalg.pinv(self.Ψ_2['-'])}
        # curvature weights Ω (2+q, 1) and Λ (2, 1)
        self.Ω = {'+': (1 / self.n) * (RZ_1['+'].T * self.𝜔['+'].T) @ (Ih['+'] * Dm)**2, '-': (1 / self.n) * (RZ_1['-'].T * self.𝜔['-'].T) @ (Ih['-'] * Dm)**2}
        self.Λ = {'+': (1 / self.n) * (self.R_1['+'].T * self.𝜔['+'].T) @ (Ih['+'] * Dm)**2, '-': (1 / self.n) * (self.R_1['-'].T * self.𝜔['-'].T) @ (Ih['-'] * Dm)**2}
        self.e_0 = torch.tensor([[1.0], [0.0]], dtype=self.dtype, device=self.device)
        self.e_2Γ = torch.tensor([[0.0], [0.0], [1.0]], dtype=self.dtype, device=self.device)
        self.e_2Ψ = torch.cat([self.e_2Γ, torch.zeros((self.q, 1), dtype=self.dtype, device=self.device)], dim=0)

        # local IV regression: 𝜈 = (α_0, α_1, 𝛾') at bandwidth h; π = (π_0, π_1, π_2, 𝛾') at bandwidth b
        self.𝜈 = {'+': self.Ψ_1_inv['+'] @ (RZ_1['+'].T * self.𝜔['+'].T) / self.n @ self.Y, '-': self.Ψ_1_inv['-'] @ (RZ_1['-'].T * self.𝜔['-'].T) / self.n @ self.Y}
        self.π = {'+': self.Ψ_2_inv['+'] @ (RZ_2['+'].T * self.𝛿['+'].T) / self.n @ self.Y, '-': self.Ψ_2_inv['-'] @ (RZ_2['-'].T * self.𝛿['-'].T) / self.n @ self.Y}
        self.𝛾 = {'+': self.𝜈['+'][2:, :], '-': self.𝜈['-'][2:, :]}  # (q, 1)
        self.Δ𝛾 = self.𝛾['+'] - self.𝛾['-']  # (q, 1)
        # local linear / quadratic regressions of W on D: β_w (2, q), κ_w (3, q)
        self.β_w = {'+': self.Γ_1_inv['+'] @ (self.R_1['+'].T * self.𝜔['+'].T) / self.n @ self.W, '-': self.Γ_1_inv['-'] @ (self.R_1['-'].T * self.𝜔['-'].T) / self.n @ self.W}
        self.κ_w = {'+': self.Γ_2_inv['+'] @ (self.R_2['+'].T * self.𝛿['+'].T) / self.n @ self.W, '-': self.Γ_2_inv['-'] @ (self.R_2['-'].T * self.𝛿['-'].T) / self.n @ self.W}
        # e_w = (1, 0, β_{+,0}^w') evaluates the IV fit at the cutoff and the right-limit of W (used on both sides)
        self.e_w = torch.cat([self.e_0, self.β_w['+'][[0], :].T], dim=0)  # (2+q, 1)
        # curvature estimates ĝ^(2)(0) = (2/b²) e_2'π, μ̂^(2)_{w_j}(0) = (2/b²) e_2'κ^{w_j}
        self.g_2 = {'+': 2 * Ib['+']**2 * self.π['+'][2, 0], '-': 2 * Ib['-']**2 * self.π['-'][2, 0]}
        self.μ_2 = {'+': 2 * Ib['+']**2 * self.κ_w['+'][[2], :], '-': 2 * Ib['-']**2 * self.κ_w['-'][[2], :]}  # (1, q)
        # bias terms η_IV = (h²/2) Ω ĝ^(2)(0) in R^{2+q}, η_w = (h²/2) Λ μ̂^(2)(0) in R^{2×q}
        self.η_IV = {'+': (self.h['+']**2 / 2) * self.Ω['+'] * self.g_2['+'], '-': (self.h['-']**2 / 2) * self.Ω['-'] * self.g_2['-']}
        self.η_w = {'+': (self.h['+']**2 / 2) * self.Λ['+'] @ self.μ_2['+'], '-': (self.h['-']**2 / 2) * self.Λ['-'] @ self.μ_2['-']}

        # Linear smoother rows (1, n): e_w' 𝜈 = (1/n) P_us @ Y, and e_w'(𝜈 - Ψ_1⁻¹ η_IV) = (1/n) P_bc @ Y,
        # since (2/b²) e_2'Ψ_2⁻¹(1/n)Σ δ_i [R_2; Z] Y_i = ĝ^(2)(0) and h²/b² = ξ²
        self.P_us = {'+': self.e_w.T @ self.Ψ_1_inv['+'] @ (RZ_1['+'].T * self.𝜔['+'].T), '-': self.e_w.T @ self.Ψ_1_inv['-'] @ (RZ_1['-'].T * self.𝜔['-'].T)}
        self.P_bc = {'+': self.P_us['+'] - self.ξ['+']**2 * (self.e_w.T @ self.Ψ_1_inv['+'] @ self.Ω['+']) * (self.e_2Ψ.T @ self.Ψ_2_inv['+'] @ (RZ_2['+'].T * self.𝛿['+'].T)),
                     '-': self.P_us['-'] - self.ξ['-']**2 * (self.e_w.T @ self.Ψ_1_inv['-'] @ self.Ω['-']) * (self.e_2Ψ.T @ self.Ψ_2_inv['-'] @ (RZ_2['-'].T * self.𝛿['-'].T))}
        # Rows (1, n) for the W_j level at the cutoff: e_0'Γ_1⁻¹(1/n)Σ ω_i R_i W_ij = (1/n) Q_us @ W_j, bias-corrected by e_0'Γ_{+,1}⁻¹ η_{+,w_j}
        self.Q_us = self.e_0.T @ self.Γ_1_inv['+'] @ (self.R_1['+'].T * self.𝜔['+'].T)
        self.Q_bc = self.Q_us - self.ξ['+']**2 * (self.e_0.T @ self.Γ_1_inv['+'] @ self.Λ['+']) * (self.e_2Γ.T @ self.Γ_2_inv['+'] @ (self.R_2['+'].T * self.𝛿['+'].T))

        # residuals from the local quadratic (bandwidth b) fits, used in the variance estimators: ε_y (n, 1), ε_w (n, q), rescaled by
        # the leverage correction selected by vce (default hc3); the leverages are the hat values of the weighted regressions of Y on
        # [R_2, W] and of W on R_2 at b (for the IV fit, the hat values of the regressor matrix, as in 2SLS, which lie in [0, 1])
        lev_IV = {sn: ((RW_2[sn] @ torch.linalg.pinv((RW_2[sn].T * self.𝛿[sn].T) @ RW_2[sn] / self.n)) * RW_2[sn]).sum(dim=1, keepdim=True) * self.𝛿[sn] / self.n for sn in ('+', '-')}
        lev_w = {sn: ((self.R_2[sn] @ self.Γ_2_inv[sn]) * self.R_2[sn]).sum(dim=1, keepdim=True) * self.𝛿[sn] / self.n for sn in ('+', '-')}
        k = {sn: int((self.𝛿[sn] > 0).sum()) for sn in ('+', '-')}
        self.resid_y = {'+': self.Y - RW_2['+'] @ self.π['+'], '-': self.Y - RW_2['-'] @ self.π['-']}
        self.ε_y = {sn: self.resid_y[sn] * torch.sqrt(hc_weights(lev_IV[sn], k[sn], 3 + self.q, self.vce)) for sn in ('+', '-')}
        self.ε_w = {sn: (self.W - self.R_2[sn] @ self.κ_w[sn]) * torch.sqrt(hc_weights(lev_w[sn], k[sn], 3, self.vce)) for sn in ('+', '-')}
        # variances of the curvature estimates ĝ^(2)(0) = (2/b²) e_2'π and μ̂^(2)_{w_j}(0), used to regularize the bias constant in the
        # bandwidth selection: Var(e_2'π) = (1/n²) e_2'Ψ_2⁻¹ (Σ δ_i² [R_2; Z]_i [R_2; Z]_i' ε²_{y,i}) Ψ_2⁻¹' e_2 (sandwich), and likewise for κ_w
        sand_IV = {sn: (self.e_2Ψ.T @ self.Ψ_2_inv[sn] @ RZ_2[sn].T) * (self.𝛿[sn] * self.ε_y[sn]).T for sn in ('+', '-')}  # (1, n)
        sand_w = {sn: (self.e_2Γ.T @ self.Γ_2_inv[sn] @ self.R_2[sn].T) * self.𝛿[sn].T for sn in ('+', '-')}  # (1, n), times ε_w below
        self.var_g_2 = {sn: (2 * Ib[sn]**2)**2 * torch.sum(sand_IV[sn]**2) / self.n**2 for sn in ('+', '-')}
        self.var_μ_2 = {sn: (2 * Ib[sn]**2)**2 * torch.sum((sand_w[sn].T * self.ε_w[sn])**2, dim=0, keepdim=True) / self.n**2 for sn in ('+', '-')}  # (1, q)
        # influence scores a_i (n, 1): the linearization ŝ'X_i of Corollary clt_decomp, written with the normalized weights ω, δ.
        # The W-level correction (𝛾_+ - 𝛾_-)'(β̂^w - β^w) is attributed to the '+' side, as in the MSE decomposition.
        get_a = lambda P, Q, sn: P[sn].T * self.ε_y[sn] + (Q.T * self.ε_w['+']) @ self.Δ𝛾 * (sn == '+')
        self.a_us = {'+': get_a(self.P_us, self.Q_us, '+'), '-': get_a(self.P_us, self.Q_us, '-')}
        self.a_bc = {'+': get_a(self.P_bc, self.Q_bc, '+'), '-': get_a(self.P_bc, self.Q_bc, '-')}
        # v² / (n h) is the variance contribution of each side: Var ≈ (1/n²) Σ a_i² = (h/n Σ a_i²) / (n h)
        self.v_rbc = {'+': torch.sqrt((self.h['+'] / self.n) * torch.sum(self.a_bc['+']**2)), '-': torch.sqrt((self.h['-'] / self.n) * torch.sum(self.a_bc['-']**2))}

    def __get_bias(self):
        # IV bias per side e_w'Ψ_1⁻¹ η_IV, and the W-level bias Σ_j (𝛾_{+,j} - 𝛾_{-,j}) e_0'Γ_{+,1}⁻¹ η_{+,w_j}
        bias_IV = {'+': (self.e_w.T @ self.Ψ_1_inv['+'] @ self.η_IV['+'])[0, 0], '-': (self.e_w.T @ self.Ψ_1_inv['-'] @ self.η_IV['-'])[0, 0]}
        bias_w = (self.e_0.T @ self.Γ_1_inv['+'] @ self.η_w['+'] @ self.Δ𝛾)[0, 0]
        return bias_IV, bias_w

    def __get_bandwidth(self):
        # Plug-in MSE bandwidth for the PDD estimator, in the structure of rdbwselect (one shot, no fixed-point iteration):
        #   0. pilot c = C_c min{sd(D), IQR/1.349} n^{-1/5};
        #   1–2. curvature bandwidth b_c: the MSE-optimal bandwidth for estimating the second derivative of the outcome regression
        #        (curvature_bandwidth, computed on Y), at which ĝ^(2)_±(0) and μ̂^(2)_{w_j}(0) are estimated;
        #   3. with the variance constants Ĉ_1^± = (c/n) Σ_i a_i² (influence scores at c) and the bias constants
        #        B̂_± = (2/c²) × (bias term of each side) (weights at c, curvatures at b_c), regularized by R_± = 3 Var(B̂_±),
        #        h_± = (Ĉ_1^±/(B̂_±² + R_±))^{1/5} n^{-1/5} for 'msetwo', or the common h = ((Ĉ_1^+ + Ĉ_1^-)/((B̂_+ - B̂_-)² + R_+ + R_-))^{1/5} n^{-1/5}
        #        for 'mse' (the minimizers of each side's AMSE and of the joint AMSE in Proposition prop:mse_decomp, respectively).
        # The bias-correction bandwidth of the estimator remains b = h / ρ.
        two = self.bwselect == 'msetwo'
        c = rot_pilot(self.D.flatten(), self.kernel_name)
        with torch.no_grad():
            self.h = {'+': c, '-': c}
            self.b = curvature_bandwidth(self.D, self.Y, self.cutoff, self.kernel, c, two)
            self.__build_matrices()
            C_1 = {sn: (self.h[sn] / self.n) * torch.sum(self.a_us[sn]**2) for sn in ('+', '-')}
            bias_IV, bias_w = self.__get_bias()
            B = {'+': (2 / self.h['+']**2) * (bias_IV['+'] + bias_w), '-': (2 / self.h['-']**2) * bias_IV['-']}
            # Imbens–Kalyanaraman regularization R_± = 3 Var(B̂_±), treating the weights e_w'Ψ_1⁻¹Ω, e_0'Γ_1⁻¹Λ and 𝛾 as fixed
            a = {sn: (self.e_w.T @ self.Ψ_1_inv[sn] @ self.Ω[sn])[0, 0] for sn in ('+', '-')}
            c_w = (self.e_0.T @ self.Γ_1_inv['+'] @ self.Λ['+'])[0, 0]
            R = {'+': 3 * (a['+']**2 * self.var_g_2['+'] + torch.sum((c_w * self.Δ𝛾.T)**2 * self.var_μ_2['+'])), '-': 3 * a['-']**2 * self.var_g_2['-']}
            if two:
                h = {sn: (C_1[sn] / torch.clamp(B[sn]**2 + R[sn], min=self.reg))**(1/5) * self.n**(-1/5) for sn in ('+', '-')}
            else:
                h_c = ((C_1['+'] + C_1['-']) / torch.clamp((B['+'] - B['-'])**2 + R['+'] + R['-'], min=self.reg))**(1/5) * self.n**(-1/5)
                h = {'+': h_c, '-': h_c}
        h_max = torch.max(torch.abs(self.D - self.cutoff))
        return {sn: torch.clamp(h[sn], max=h_max) for sn in ('+', '-')}, True

    def fit(self):
        if type(self.custom_bandwidth) != type(None):
            self.h = {'-': self.custom_bandwidth[0], '+': self.custom_bandwidth[1]}
            status = True
        else:
            self.h, status = self.__get_bandwidth()
        h_max = torch.max(torch.abs(self.D - self.cutoff))
        for sn in ('+', '-'):
            if self.h[sn] >= h_max * (1 - 1e-6):
                warnings.warn(f"Bandwidth h{sn} reached the range of the data.")
        self.b = {'+': 1/self.ρ * self.h['+'], '-': 1/self.ρ * self.h['-']}

        self.__build_matrices()
        bias_IV, bias_w = self.__get_bias()
        # τ̂_pdd = e_w'𝜈_+ - e_w'𝜈_-, then subtract the estimated bias (Equation pdd_rbc_def); the W correction is attributed to the '+' side
        est_pos = (self.e_w.T @ self.𝜈['+'])[0, 0] - bias_IV['+'] - bias_w
        est_neg = (self.e_w.T @ self.𝜈['-'])[0, 0] - bias_IV['-']
        est = est_pos - est_neg
        se = torch.sqrt(self.v_rbc['+']**2/(self.n * self.h['+']) + self.v_rbc['-']**2/(self.n * self.h['-']))
        se_pos = torch.sqrt(self.v_rbc['+']**2/(self.n * self.h['+']))
        se_neg = torch.sqrt(self.v_rbc['-']**2/(self.n * self.h['-']))

        resids = (self.ind['+'] * self.resid_y['+'] + self.ind['-'] * self.resid_y['-']).flatten().detach().cpu().numpy()
        def predict(d) -> np.ndarray:
            # bias-corrected fitted outcome at d, holding W at its right-limit β̂_{+,0}^w
            d = torch.as_tensor(d, dtype=self.dtype, device=self.device)
            if d.ndim == 1: d = d.reshape(-1, 1)
            Ih, dm, one_m = {'+': 1 / self.h['+'], '-': 1 / self.h['-']}, d - self.cutoff, torch.ones((d.shape[0], 1), dtype=self.dtype, device=self.device)
            ind = {'+': dm >= 0, '-': dm < 0}
            β_w0 = self.β_w['+'][[0], :].expand(d.shape[0], -1)
            r = {'+': torch.cat([one_m, Ih['+'] * dm, β_w0], dim=1),
                 '-': torch.cat([one_m, Ih['-'] * dm, β_w0], dim=1)}
            Yhat = {'+': r['+'] @ (self.𝜈['+'] - self.Ψ_1_inv['+'] @ self.η_IV['+']),
                    '-': r['-'] @ (self.𝜈['-'] - self.Ψ_1_inv['-'] @ self.η_IV['-'])}
            pred = ind['+'] * Yhat['+'] + ind['-'] * Yhat['-']
            return pred.flatten().detach().cpu().numpy()

        res = Results(model = 'Placebo Discontinuity Design',
                    est = est.item(),
                    est_pos = est_pos.item(),
                    est_neg = est_neg.item(),
                    se = se.item(),
                    se_pos = se_pos.item(),
                    se_neg = se_neg.item(),
                    resid = resids,
                    bandwidth = {'+': self.h['+'].item(), '-': self.h['-'].item()},
                    n = self.n,
                    predict = predict,
                    status = status)
        return res

class rdd:
    def __init__(self, Y: np.ndarray, D: np.ndarray, cutoff=0.0, alpha=0.05, kernel='triangle', 
                 bandwidth = None, bwselect = 'msetwo', vce = 'hc3', dtype = torch.float64, device = 'cpu', seed = 10042002):
        self.dtype, self.device = dtype, device
        self.Y = torch.as_tensor(Y, dtype=dtype, device=device)
        if self.Y.ndim == 1: self.Y = self.Y.reshape(-1, 1)
        self.D = torch.as_tensor(D, dtype=dtype, device=device)
        if self.D.ndim == 1: self.D = self.D.reshape(-1, 1)
        self.n = int(self.D.shape[0])
        self.cutoff = torch.tensor(cutoff, dtype=dtype, device=device)
        self.alpha = torch.tensor(alpha, dtype=dtype, device=device)
        self.kernel_name = kernel if kernel in ('triangle', 'rectangle') else 'epanechnikov'
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
        # bwselect: 'mserd'/'cerrd' use a common bandwidth, 'msetwo'/'certwo' one per side; 'cer*' scales the MSE-optimal bandwidth by n^{-1/20}
        if bwselect not in ('mserd', 'cerrd', 'msetwo', 'certwo'):
            raise ValueError("bwselect must be one of 'mserd', 'cerrd', 'msetwo', 'certwo'")
        self.bwselect, self.vce = bwselect, vce
        self.h = {'-': 2 * torch.std(self.D), '+': 2 * torch.std(self.D)}
        self.b = {'+': 1/self.ρ * self.h['+'], '-': 1/self.ρ * self.h['-']}

    def __build_matrices(self):
        one_n = torch.ones((self.n, 1), dtype=self.dtype, device=self.device)
        Ih, Ib, Dm = {'+': 1 / self.h['+'], '-': 1 / self.h['-']}, {'+': 1 / self.b['+'], '-': 1 / self.b['-']}, self.D - self.cutoff
        self.R_1 = {'+': torch.cat([one_n, Ih['+'] * Dm], dim=1), '-': torch.cat([one_n, Ih['-'] * Dm], dim=1)}
        self.R_2 = {'+': torch.cat([one_n, Ib['+'] * Dm, (Ib['+'] * Dm)**2], dim=1), '-': torch.cat([one_n, Ib['-'] * Dm, (Ib['-'] * Dm)**2], dim=1)}

        self.ind = {'+': (self.D >= self.cutoff), '-': (self.D < self.cutoff)}
        # nan_to_num: Inf*0=NaN when 1/h overflows for tiny h; kernel should return 0 there
        self.𝜔 = {'+': torch.nan_to_num(Ih['+'] * self.ind['+'] * self.kernel(Ih['+'] * Dm)),
                  '-': torch.nan_to_num(Ih['-'] * self.ind['-'] * self.kernel(Ih['-'] * Dm))}
        self.𝛿 = {'+': torch.nan_to_num(Ib['+'] * self.ind['+'] * self.kernel(Ib['+'] * Dm)),
                  '-': torch.nan_to_num(Ib['-'] * self.ind['-'] * self.kernel(Ib['-'] * Dm))}

        self.Γ_1 = {'+': (1 / self.n) * (self.R_1['+'].T * self.𝜔['+'].T) @ self.R_1['+'], '-': (1 / self.n) * (self.R_1['-'].T * self.𝜔['-'].T) @ self.R_1['-']}
        self.Γ_2 = {'+': (1 / self.n) * (self.R_2['+'].T * self.𝛿['+'].T) @ self.R_2['+'], '-': (1 / self.n) * (self.R_2['-'].T * self.𝛿['-'].T) @ self.R_2['-']}
        self.Γ_1_inv = {'+': torch.linalg.pinv(self.Γ_1['+']), '-': torch.linalg.pinv(self.Γ_1['-'])}
        self.Γ_2_inv = {'+': torch.linalg.pinv(self.Γ_2['+']), '-': torch.linalg.pinv(self.Γ_2['-'])}
        self.Λ_1 = {'+': (1 / self.n) * (self.R_1['+'].T * self.𝜔['+'].T) @ (Ih['+'] * Dm)**2, '-': (1 / self.n) * (self.R_1['-'].T * self.𝜔['-'].T) @ (Ih['-'] * Dm)**2}
        self.Λ_2 = {'+': (1 / self.n) * (self.R_2['+'].T * self.𝛿['+'].T) @ (Ib['+'] * Dm)**2, '-': (1 / self.n) * (self.R_2['-'].T * self.𝛿['-'].T) @ (Ib['-'] * Dm)**2}
        self.e_0 = torch.tensor([[1.0], [0.0]], dtype=self.dtype, device=self.device)
        self.e_2 = torch.tensor([[0.0], [0.0], [1.0]], dtype=self.dtype, device=self.device)

        self.B_2β = {'+': self.Γ_2_inv['+'] @ (self.R_2['+'].T * self.𝛿['+'].T) / self.n @ self.Y,
                  '-': self.Γ_2_inv['-'] @ (self.R_2['-'].T * self.𝛿['-'].T) / self.n @ self.Y}
        self.H_1β = {'+': self.Γ_1_inv['+'] @ (self.R_1['+'].T * self.𝜔['+'].T) / self.n @ self.Y,
                  '-': self.Γ_1_inv['-'] @ (self.R_1['-'].T * self.𝜔['-'].T) / self.n @ self.Y}
        self.ε = {'+': (self.Y - self.R_1['+'] @ self.H_1β['+']),  # (n, 1)
                  '-': (self.Y - self.R_1['-'] @ self.H_1β['-'])}  # (n, 1)
        # σ²_i from the residuals of the local quadratic fit, with the leverage correction selected by vce (default hc3)
        get_σ = lambda sn: (self.Y - self.R_2[sn] @ self.B_2β[sn]).abs() * torch.sqrt(hc_weights(
            ((self.R_2[sn] @ self.Γ_2_inv[sn]) * self.R_2[sn]).sum(dim=1, keepdim=True) * self.𝛿[sn] / self.n, int((self.𝛿[sn] > 0).sum()), 3, self.vce))
        self.σ = {'+': get_σ('+'), '-': get_σ('-')}  # (n, 1)
        self.P_bc = {'+': self.Γ_1_inv['+'] @ (self.R_1['+'].T * self.𝜔['+'].T) - (self.h['+'] / self.b['+'])**2 * self.Γ_1_inv['+'] @ self.Λ_1['+'] @ self.e_2.T @ self.Γ_2_inv['+'] @ (self.R_2['+'].T * self.𝛿['+'].T),
                     '-': self.Γ_1_inv['-'] @ (self.R_1['-'].T * self.𝜔['-'].T) - (self.h['-'] / self.b['-'])**2 * self.Γ_1_inv['-'] @ self.Λ_1['-'] @ self.e_2.T @ self.Γ_2_inv['-'] @ (self.R_2['-'].T * self.𝛿['-'].T)}
        # e_0.T @ P @ diag(σ²) @ P.T @ e_0 = ||P[0,:] ⊙ σ||² — avoids materialising n×n Σ
        self.v_rbc = {'+': torch.sqrt((self.h['+'] / self.n) * torch.sum(self.P_bc['+'][0, :]**2 * self.σ['+'].flatten()**2)),
                      '-': torch.sqrt((self.h['-'] / self.n) * torch.sum(self.P_bc['-'][0, :]**2 * self.σ['-'].flatten()**2))}

    def __get_bandwidth(self):
        # Plug-in bandwidth selection following rdbwselect (Calonico, Cattaneo and Titiunik, 2014) for the local linear (p = 1)
        # estimator of μ_±(0) bias-corrected by a local quadratic (q = 2) fit:
        #   0. pilot c = C_c min{sd(D), IQR/1.349} n^{-1/5};
        #   1–2. b: MSE-optimal bandwidth for the local quadratic estimator of μ^(2)(0) (see curvature_bandwidth);
        #   3. h: MSE-optimal bandwidth for the local linear estimator of μ(0), with μ^(2)(0) from a quadratic fit at b;
        #   4. for coverage-error optimality, h_cer = h_mse n^{-1/20} (Calonico, Cattaneo and Farrell, 2018).
        # The b of steps 1–2 is only used to estimate the curvature; the bias-correction bandwidth of the estimator remains b = h / ρ.
        two = self.bwselect in ('msetwo', 'certwo')
        c = rot_pilot(self.D.flatten(), self.kernel_name)
        b = curvature_bandwidth(self.D, self.Y, self.cutoff, self.kernel, c, two)
        h = mse_bandwidth(self.D, self.Y, self.cutoff, self.kernel, p=1, ν=0, h_V={'+': c, '-': c}, h_B=b, two=two)
        if self.bwselect in ('cerrd', 'certwo'):
            h = {sn: h[sn] * self.n**(-1/20) for sn in ('+', '-')}
        h_max = torch.max(torch.abs(self.D - self.cutoff))
        return {sn: torch.clamp(h[sn], max=h_max) for sn in ('+', '-')}, True

    def fit(self):
        if type(self.custom_bandwidth) != type(None):
            self.h = {'-': self.custom_bandwidth[0], '+': self.custom_bandwidth[1]}
            status = True
        else:
            self.h, status = self.__get_bandwidth()
        self.b = {'+': 1/self.ρ * self.h['+'], '-': 1/self.ρ * self.h['-']}
        
        self.__build_matrices()
        P_bc = self.P_bc['+'] - self.P_bc['-']
        est = (1/self.n) * self.e_0.T @ P_bc @ self.Y
        est_pos = (1/self.n) * self.e_0.T @ self.P_bc['+'] @ self.Y
        est_neg = (1/self.n) * self.e_0.T @ self.P_bc['-'] @ self.Y
        
        se = torch.sqrt(self.v_rbc['+']**2/(self.n * self.h['+']) + self.v_rbc['-']**2/(self.n * self.h['-']))
        se_pos = torch.sqrt(self.v_rbc['+']**2/(self.n * self.h['+']))
        se_neg = torch.sqrt(self.v_rbc['-']**2/(self.n * self.h['-']))
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
                    est_pos = est_pos.item(),
                    est_neg = est_neg.item(),
                    se = se.item(),
                    se_pos = se_pos.item(),
                    se_neg = se_neg.item(),
                    resid = resids,
                    bandwidth = {'+': self.h['+'].item(), '-': self.h['-'].item()},
                    n = self.n,
                    predict = predict,
                    status = status)
        return res
    