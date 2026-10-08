import numpy as np, warnings, torch, sys, os, re
from scipy.stats import norm
from scipy.optimize import minimize as scipy_minimize
sys.path.append('/'.join(re.split('/|\\\\', os.path.dirname( __file__ ))[0:-1]))
from rddesign.helpers import *
from math import factorial, log

class pdd:
    def __init__(self, Y: np.ndarray, W: np.ndarray, D: np.ndarray, Z: np.ndarray, cutoff=0.0, alpha=0.05, kernel='triangle',
                 bandwidth = None, bwselect = 'msetwo', pilot = None, dtype = torch.float64, device = 'cpu', tol = 1e-3, max_iter = 50, reg = 1e-10, damp = 0.5):
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
        # pilot bandwidth h^(0) for the ROT iteration (default 2 sd(D)); b = h / ρ throughout
        h_0 = 2 * torch.std(self.D) if pilot is None else torch.as_tensor(pilot, dtype=dtype, device=device)
        self.h = {'+': h_0.clone(), '-': h_0.clone()}
        self.b = {'+': 1/self.ρ * self.h['+'], '-': 1/self.ρ * self.h['-']}
        # ROT bandwidth controls: 'msetwo' picks (h_+, h_-) minimizing each side's AMSE, 'mse' a common h minimizing the joint AMSE;
        # tolerance ε_tol, maximum iterations T, regularization ε_reg for the squared bias constant, damping λ ∈ (0, 1] of the fixed-point update
        if bwselect not in ('mse', 'msetwo'):
            raise ValueError("bwselect must be 'mse' or 'msetwo'")
        self.bwselect, self.tol, self.max_iter, self.reg, self.damp = bwselect, tol, max_iter, reg, damp

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

        # residuals from the local quadratic (bandwidth b) fits, used in the variance estimators: ε_y (n, 1), ε_w (n, q)
        self.ε_y = {'+': self.Y - RW_2['+'] @ self.π['+'], '-': self.Y - RW_2['-'] @ self.π['-']}
        self.ε_w = {'+': self.W - self.R_2['+'] @ self.κ_w['+'], '-': self.W - self.R_2['-'] @ self.κ_w['-']}
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

    def __rot_update(self, h):
        # One step of the ROT rule at bandwidths h = {'+': h_+, '-': h_-}. With side-specific variance constants Ĉ_1^± = (h_±/n) Σ_i a_i²
        # and bias constants B̂_± = (2/h_±²) × (bias term of each side), the AMSE is Ĉ_1^+/(n h_+) + Ĉ_1^-/(n h_-) + (h_+² B̂_+ - h_-² B̂_-)²/4.
        with torch.no_grad():
            self.h = dict(h)
            self.b = {'+': 1/self.ρ * self.h['+'], '-': 1/self.ρ * self.h['-']}
            self.__build_matrices()
            C_1 = {'+': (self.h['+'] / self.n) * torch.sum(self.a_us['+']**2), '-': (self.h['-'] / self.n) * torch.sum(self.a_us['-']**2)}
            bias_IV, bias_w = self.__get_bias()
            B = {'+': (2 / self.h['+']**2) * (bias_IV['+'] + bias_w), '-': (2 / self.h['-']**2) * bias_IV['-']}
            h_max = torch.max(torch.abs(self.D - self.cutoff))
            if self.bwselect == 'mse':
                # common bandwidth: AMSE(h) = (Ĉ_1^+ + Ĉ_1^-)/(n h) + (h⁴/4)(B̂_+ - B̂_-)², minimized at h = ((Ĉ_1^+ + Ĉ_1^-)/(B̂_+ - B̂_-)²)^{1/5} n^{-1/5}
                h_new = torch.clamp(((C_1['+'] + C_1['-']) / torch.clamp((B['+'] - B['-'])**2, min=self.reg))**(1/5) * self.n**(-1/5), max=h_max)
                return {'+': h_new, '-': h_new}
            else:
                # side-specific bandwidths: each side's AMSE_±(h) = Ĉ_1^±/(n h) + (h⁴/4) B̂_±² is minimized at h_± = (Ĉ_1^±/B̂_±²)^{1/5} n^{-1/5}
                # (the joint AMSE is not minimized, since it can be driven to zero by cancelling the two biases when they share a sign)
                return {sn: torch.clamp((C_1[sn] / torch.clamp(B[sn]**2, min=self.reg))**(1/5) * self.n**(-1/5), max=h_max) for sn in ('+', '-')}

    def __get_bandwidth(self):
        # ROT MSE bandwidth (Algorithm alg:mse_bandwidth): iterate h ← Φ(h) from the pilot bandwidth to a fixed point.
        # The update is damped in log scale, log h ← (1 - λ) log h + λ log Φ(h), since the undamped map can cycle; if the
        # iteration has not settled, λ is halved and the iteration continues from the last iterate.
        h, λ, converged = dict(self.h), self.damp, False
        for attempt in range(4):
            for t in range(self.max_iter):
                h_rot = self.__rot_update(h)
                h_new = {sn: torch.exp((1 - λ) * torch.log(h[sn]) + λ * torch.log(h_rot[sn])) for sn in ('+', '-')}
                if max(torch.abs(h_new[sn] - h[sn]) for sn in ('+', '-')) <= self.tol:
                    h, converged = h_new, True
                    break
                h = h_new
            if converged:
                break
            λ = λ / 2
        return h, converged

    def fit(self):
        if type(self.custom_bandwidth) != type(None):
            self.h = {'-': self.custom_bandwidth[0], '+': self.custom_bandwidth[1]}
            status = True
        else:
            self.h, status = self.__get_bandwidth()
        if not status:
            warnings.warn('Bandwidth iteration did not converge.')
        h_max = torch.max(torch.abs(self.D - self.cutoff))
        for sn in ('+', '-'):
            if self.h[sn] >= h_max * (1 - 1e-6):
                warnings.warn(f"Bandwidth h{sn} reached the range of the data; the estimated curvature on that side is near zero. Consider a smaller pilot bandwidth.")
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

        resids = (self.ind['+'] * self.ε_y['+'] + self.ind['-'] * self.ε_y['-']).flatten().detach().cpu().numpy()
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
                 bandwidth = None, dtype = torch.float64, device = 'cpu', seed = 10042002):
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
        self.h = {'-': 2 * torch.std(self.D), '+': 2 * torch.std(self.D)}
        self.logh = {'-': torch.log(self.h['-']), '+': torch.log(self.h['+'])}
        self.b = {'+': 1/self.ρ * self.h['+'], '-': 1/self.ρ * self.h['-']}
        self.gen = torch.Generator(device = device).manual_seed(seed)
        self.M = self.n * int(log(self.n))
        self.I, self.J = self.__sample_perms(self.n, self.M, self.device, self.gen)

    def __sample_perms(self, n: int, nsamples: int, device = 'cpu', gen = torch.Generator()) -> torch.Tensor:
        # random set of permutations when computing mean of edgeworth terms
        N = n * (n - 1)
        nsamples = min(nsamples, N)

        k = torch.randperm(N, device = device, dtype = torch.int64, generator = gen)[:nsamples] 
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
                  '-': (self.Y - self.R_2['-'] @ self.B_2β['-']).abs()}  # (n, 1)
        self.P_bc = {'+': self.Γ_1_inv['+'] @ (self.R_1['+'].T * self.𝜔['+'].T) - (self.h['+'] / self.b['+'])**2 * self.Γ_1_inv['+'] @ self.Λ_1['+'] @ self.e_2.T @ self.Γ_2_inv['+'] @ (self.R_2['+'].T * self.𝛿['+'].T),
                     '-': self.Γ_1_inv['-'] @ (self.R_1['-'].T * self.𝜔['-'].T) - (self.h['-'] / self.b['-'])**2 * self.Γ_1_inv['-'] @ self.Λ_1['-'] @ self.e_2.T @ self.Γ_2_inv['-'] @ (self.R_2['-'].T * self.𝛿['-'].T)}
        # e_0.T @ P @ diag(σ²) @ P.T @ e_0 = ||P[0,:] ⊙ σ||² — avoids materialising n×n Σ
        self.v_rbc = {'+': torch.sqrt((self.h['+'] / self.n) * torch.sum(self.P_bc['+'][0, :]**2 * self.σ['+'].flatten()**2)),
                      '-': torch.sqrt((self.h['-'] / self.n) * torch.sum(self.P_bc['-'][0, :]**2 * self.σ['-'].flatten()**2))}

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
            (1/self.n) * self.e_0.T @ self.Γ_1_inv[sn] @ (self.Λ_1_2[sn] - self.Λ_1[sn] @ self.e_2.T @ self.Γ_2_inv[sn] @ self.Λ_2_1[sn])
        return 𝜂_bc[0, 0]
    
    def __get_bandwidth(self, tol = 0.001):
        def obj(logh_np):
            with torch.no_grad():
                logh = torch.as_tensor(logh_np, dtype=self.dtype, device=self.device)
                self.h['-'] = torch.exp(logh[0])
                self.h['+'] = torch.exp(logh[1])
                self.b = {'+': 1/self.ρ * self.h['+'], '-': 1/self.ρ * self.h['-']}
                self.__build_matrices()
                self.__build_edgeworth_terms()
                self.𝜇_3 = {'+': self.__get_𝜇_3('+'), '-': self.__get_𝜇_3('-')}
                𝜂_bc = {'+': self.__get_𝜂_bc('+'), '-': self.__get_𝜂_bc('-')}
                q_1 = {'+': self.__get_q_1('+'), '-': self.__get_q_1('-')}
                q_2 = {'+': self.__get_q_2('+'), '-': self.__get_q_2('-')}
                q_3 = {'+': self.__get_q_3('+'), '-': self.__get_q_3('-')}
                loss = (( (1/(self.n * self.h['+'])) * q_1['+'] + self.n * self.h['+']**7 * 𝜂_bc['+']**2 * q_2['+'] + self.h['+']**3 * 𝜂_bc['+'] * q_3['+'] )/self.n**(6/4))**2 +\
                    (( (1/(self.n * self.h['-'])) * q_1['-'] + self.n * self.h['-']**7 * 𝜂_bc['-']**2 * q_2['-'] + self.h['-']**3 * 𝜂_bc['-'] * q_3['-'] )/self.n**(6/4))**2
            return float(loss)

        logh0 = np.array([self.logh['-'].item(), self.logh['+'].item()])
        margin = np.log(2 * np.std(self.D.detach().cpu().numpy()))
        bounds = [(logh0[0] - abs(margin), logh0[0] + abs(margin)), (logh0[1] - abs(margin), logh0[1] + abs(margin))]
        res = scipy_minimize(obj, logh0, method='L-BFGS-B', jac='2-point', bounds=bounds,
                             options={'ftol': tol**(2), 'gtol': tol, 'maxiter': 500, 'eps': 1e-4})
        # nit<=1 means L-BFGS-B quit at the starting point without iterating; fall through to Nelder-Mead
        if not res.success or res.nit <= 1:
            res = scipy_minimize(obj, logh0, method='Nelder-Mead',
                                 options={'xatol': tol, 'fatol': tol**(2), 'maxiter': 500})
        res.x = torch.as_tensor(res.x, dtype=self.dtype, device=self.device)
        return res

    def fit(self):
        if type(self.custom_bandwidth) != type(None):
            self.h = {'-': self.custom_bandwidth[0], '+': self.custom_bandwidth[1]}
            status = True
        else:
            bres = self.__get_bandwidth()
            self.h = {'-': torch.exp(bres.x[0]), '+': torch.exp(bres.x[1])}
            status = bres.success
        if not status:
            warnings.warn('Bandwidth optimization did not converge.')
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
    