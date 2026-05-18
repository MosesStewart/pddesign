import numpy as np, pandas as pd, torch, sys, os, re
from matplotlib import pyplot as plt
sys.path.append('/'.join(re.split('/|\\\\', os.path.dirname( __file__ ))[0:-2]))
from rddesign.main import *
from analysis.almond.results import _add_side_brackets, plot_res, plot_rdd

TINY_SIZE = 14
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=TINY_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE + 2)  # fontsize of the figure title
plt.rcParams["font.family"] = "Times New Roman"

def main():
    indir = 'output/derived/almond'
    outdir = 'output/analysis/almond'
    df = pd.read_csv(f'{indir}/clean_data_90s.csv')
    Y = df.loc[:, 'death'].values
    D = df.loc[:, 'brthwgt'].values
    W = df.loc[:, 'meduc'].values
    Z = df.loc[:, 'wknd'].values
    
    with open(f'{outdir}/summary_nobc.txt', 'w') as summary:
        model = rdd_nobc(Y, D, cutoff = 1500, kernel = 'uniform', bandwidth = [85, 85])
        res_rdd = model.fit()
        summary.write('\nY ~ Death')
        summary.write(str(res_rdd))
        
        fig, ax = plot_rdd(res_rdd, Y, D)
        ax.set_ylabel('$\\mathbb{E}\\left[Y \\mid D = d\\right]$')
        ax.set_ylim(0, 0.08)
        fig.savefig(f'{outdir}/rdd_90s_nobc.pdf', transparent = True, bbox_inches="tight")

class rdd_nobc:
    def __init__(self, Y: np.ndarray, D: np.ndarray, cutoff=0.0, alpha=0.05, kernel='triangle', 
                 bandwidth = None, dtype = torch.float32, device = 'cpu'):
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
        elif kernel == 'rectangle': 
            self.kernel = rectangle_kernel
        else: 
            self.kernel = epanechnikov_kernel
        if type(bandwidth) != type(None):
            self.custom_bandwidth = torch.as_tensor(bandwidth, dtype=dtype, device=device).flatten()
        else:
            self.custom_bandwidth = None
        self.h = {'-': 3 * torch.std(self.D) * self.n**(-1/4), '+': 3 * torch.std(self.D) * self.n**(-1/4)}

    def __build_matrices(self):
        one_n = torch.ones((self.n, 1), dtype=self.dtype, device=self.device)
        Ih, Dm = {'+': 1 / self.h['+'], '-': 1 / self.h['-']}, self.D - self.cutoff
        self.R_1 = {'+': torch.cat([one_n, Ih['+'] * Dm], dim=1), '-': torch.cat([one_n, Ih['-'] * Dm], dim=1)}

        self.ind = {'+': (self.D >= self.cutoff), '-': (self.D < self.cutoff)}
        self.𝜔 = {'+': (Ih['+'] * self.ind['+'] * self.kernel(Ih['+'] * Dm)), '-': (Ih['-'] * self.ind['-'] * self.kernel(Ih['-'] * Dm))}
        
        self.I_n = torch.eye(self.n, dtype=self.dtype, device=self.device)
        self.Γ_1 = {'+': (1 / self.n) * (self.R_1['+'].T * self.𝜔['+'].T) @ self.R_1['+'], '-': (1 / self.n) * (self.R_1['-'].T * self.𝜔['-'].T) @ self.R_1['-']}
        self.Γ_1_inv = {'+': torch.linalg.pinv(self.Γ_1['+']), '-': torch.linalg.pinv(self.Γ_1['-'])}
        self.e_0 = torch.tensor([[1.0], [0.0]], dtype=self.dtype, device=self.device)

        self.H_1β = {'+': self.Γ_1_inv['+'] @ (self.R_1['+'].T * self.𝜔['+'].T) / self.n @ self.Y, 
                  '-': self.Γ_1_inv['-'] @ (self.R_1['-'].T * self.𝜔['-'].T) / self.n @ self.Y}
        self.ε = {'+': (self.Y - self.R_1['+'] @ self.H_1β['+']),  # (n, 1)
                  '-': (self.Y - self.R_1['-'] @ self.H_1β['-'])}  # (n, 1)
        self.σ = {'+': (self.Y - self.R_1['+'] @ self.H_1β['+']).abs(),  # (n, 1)
                  '-': (self.Y - self.R_1['-'] @ self.H_1β['-']).abs(),}  # (n, 1)
        self.Σ = {'+': torch.diag(self.σ['+'].flatten()**2),
                  '-': torch.diag(self.σ['-'].flatten()**2)}
        self.P = {'+': self.Γ_1_inv['+'] @ (self.R_1['+'].T * self.𝜔['+'].T), 
                     '-': self.Γ_1_inv['-'] @ (self.R_1['-'].T * self.𝜔['-'].T)}
        self.v_uc = {'+': torch.sqrt((self.h['+'] / self.n) * (self.e_0.T @ self.P['+'] @ self.Σ['+'] @ self.P['+'].T @ self.e_0)),
                      '-': torch.sqrt((self.h['-'] / self.n) * (self.e_0.T @ self.P['-'] @ self.Σ['-'] @ self.P['-'].T @ self.e_0))}
        
    def fit(self):
        if type(self.custom_bandwidth) != type(None):
            self.h = {'-': self.custom_bandwidth[0], '+': self.custom_bandwidth[1]}

        self.__build_matrices()
        P_bc = self.P['+'] - self.P['-']
        est = (1/self.n) * self.e_0.T @ P_bc @ self.Y
        est_pos = (1/self.n) * self.e_0.T @ self.P['+'] @ self.Y
        est_neg = (1/self.n) * self.e_0.T @ self.P['-'] @ self.Y
        
        se = torch.sqrt(self.v_uc['+']**2/(self.n * self.h['+']) + self.v_uc['-']**2/(self.n * self.h['-']))
        se_pos = torch.sqrt(self.v_uc['+']**2/(self.n * self.h['+']))
        se_neg = torch.sqrt(self.v_uc['-']**2/(self.n * self.h['-']))
        resid_pos = self.Y - self.R_1['+'] @ self.Γ_1_inv['+'] @ (self.R_1['+'].T * self.𝜔['+'].T) / self.n @ self.Y
        resid_neg = self.Y - self.R_1['-'] @ self.Γ_1_inv['-'] @ (self.R_1['-'].T * self.𝜔['-'].T) / self.n @ self.Y
        resids = (self.ind['+'] * resid_pos + self.ind['-'] * resid_neg).flatten().detach().cpu().numpy()
        def predict(d) -> np.ndarray:
            d = torch.as_tensor(d, dtype=self.dtype, device=self.device)
            if d.ndim == 1: d = d.reshape(-1, 1)
            Ih, dm, one_m = {'+': 1 / self.h['+'], '-': 1 / self.h['-']}, d - self.cutoff, torch.ones((d.shape[0], 1), dtype=self.dtype, device=self.device)
            ind = {'+': dm >= 0, '-': dm < 0}
            r = {'+': torch.cat([one_m, Ih['+'] * dm], dim=1),
                '-': torch.cat([one_m, Ih['-'] * dm], dim=1)}
            Yhat = {'+': (1/self.n) * r['+'] @ self.P['+'] @ self.Y, 
                    '-': (1/self.n) * r['-'] @ self.P['-'] @ self.Y}
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
                    status = True)
        return res

if __name__ == '__main__':
    main()