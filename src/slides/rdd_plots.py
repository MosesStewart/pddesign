import torch, pandas as pd, numpy as np, sys, os, re
from matplotlib import pyplot as plt
sys.path.append('/'.join(re.split('/|\\\\', os.path.dirname( __file__ ))[0:-1]))
from rddesign.main import *
from derived.simulation import *

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams["font.family"] = "Times New Roman"

model_s = lambda d: (d < 0) * (1/7) + (d >= 0) * (-1/7) + d/4

def sim_s(μx, ndraws = 4000, seed = 10042002):
    gen = torch.Generator().manual_seed(seed)
    U = torch.bernoulli(0.45 * torch.ones((ndraws, 1)), generator = gen)
    V = torch.bernoulli(0.90 * torch.ones((ndraws, 1)), generator = gen)
    
    D =  (U == 1) * ( (V == 1) * torch.log(torch.rand((ndraws, 1), generator = gen))/4 + (V == 0) * (torch.randn((ndraws, 1), generator = gen)/4 - 1/4) ) +\
         (U == 0) * ( (torch.randn((ndraws, 1), generator = gen)/4 + 1/4) )
    Z = 3*U/5 + torch.randn((ndraws, 1), generator = gen)/5 + D/10
    
    W = -3*U/5 + torch.randn((ndraws, 1), generator = gen)/5
    Y = μx(D) - 2*U/5 + torch.randn((ndraws, 1), generator = gen)/5
    return Y.flatten().detach().cpu().numpy(), W.flatten().detach().cpu().numpy(), D.flatten().detach().cpu().numpy(), Z.flatten().detach().cpu().numpy(), U.flatten().detach().cpu().numpy()

def main():
    outdir = 'output/slides'
    Y, W, D, Z, U = sim_s(model_s, ndraws = 4000, seed = 5)
    
    model = rdd(Y, D, cutoff = 0.0, device = 'cuda', kernel = 'triangle', bandwidth = [0.5, 0.5])
    res_rdd = model.fit()
    print(res_rdd)
    
    model = pdd(Y, W, D, Z, cutoff = 0.0, device = 'cuda', kernel = 'triangle', bandwidth = [0.5, 0.5])
    res_pdd = model.fit()
    print(res_pdd)
    
    fig, ax = plot_res(res_rdd, res_pdd, Y, D, model_s)    
    fig.savefig(f'{outdir}/full_plot.pdf', transparent = True, bbox_inches="tight")
    
    fig, ax = plot_rdd(res_rdd, Y, D)    
    ax.set_ylabel('$\\hat{\\mu}_{0, y}(d) = \\hat{\\mathbb{E}}[Y \\mid D = d]$')
    fig.savefig(f'{outdir}/rdd_plot.pdf', transparent = True, bbox_inches="tight")
    
    model = rdd(W, D, cutoff = 0.0, device = 'cuda', kernel = 'triangle', bandwidth = [0.5, 0.5])
    res_w_rdd = model.fit()
    
    fig, ax = plot_rdd(res_w_rdd, W, D)    
    ax.set_ylabel('$\\hat{\\mu}_{0, w}(d) = \\hat{\\mathbb{E}}[W \\mid D = d]$')
    fig.savefig(f'{outdir}/cont_test.pdf', transparent = True, bbox_inches="tight")

def plot_res(rres, pres, Y, D, fn):
    x1 = np.linspace(-1.5, -0.001, 200)
    x2 = np.linspace(0, 1.5, 200)
    fig, ax = plt.subplots()
    ax.scatter(D, Y, s = 5, c = "#F0F0F0")
    ax.plot(x1, fn(x1), color = "#000000", label = 'True', linewidth = 2)
    ax.plot(x1, rres.predict(x1), color = "#8DD2DB", label = 'RDD', linewidth = 2)
    ax.plot(x1, rres.predict(x1) + rres.left_ci - rres.est, color = "#8DD2DB", linewidth = 1, alpha = 0.55)
    ax.plot(x1, rres.predict(x1) + rres.right_ci - rres.est, color = '#8DD2DB', linewidth = 1, alpha = 0.55)
    ax.plot(x1, pres.predict(x1), color = '#FF6961', label = 'PDD', linewidth = 2)
    ax.plot(x1, pres.predict(x1) + pres.left_ci - pres.est, color = '#FF6961', linewidth = 1, alpha = 0.55)
    ax.plot(x1, pres.predict(x1) + pres.right_ci - pres.est, color = '#FF6961', linewidth = 1, alpha = 0.55)
    ax.plot(x2, fn(x2), color = '#000000', linewidth = 2)
    ax.plot(x2, rres.predict(x2), color = '#8DD2DB', linewidth = 2)
    ax.plot(x2, rres.predict(x2) + rres.left_ci - rres.est, color = '#8DD2DB', linewidth = 1, alpha = 0.55)
    ax.plot(x2, rres.predict(x2) + rres.right_ci - rres.est, color = '#8DD2DB', linewidth = 1, alpha = 0.55)
    ax.plot(x2, pres.predict(x2), color = '#FF6961', linewidth = 2)
    ax.plot(x2, pres.predict(x2) + pres.left_ci - pres.est, color = '#FF6961', linewidth = 1, alpha = 0.55)
    ax.plot(x2, pres.predict(x2) + pres.right_ci - pres.est, color = '#FF6961', linewidth = 1, alpha = 0.55)
    ax.vlines(x = 0, ymin=-1.5, ymax=1.5, color='#000000', alpha = 0.5, linestyle = (0, (5, 5)))
    ax.legend(loc = 'upper left')
    ax.set_xlabel('D')
    ax.set_ylabel('$\\tilde{\\mu}_{0}(D)$')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.8, 0.8)
    ax.spines[['right', 'top']].set_visible(False)
    return fig, ax

def plot_rdd(res, Y, D):
    x1 = np.linspace(-1.5, -0.001, 200)
    x2 = np.linspace(0, 1.5, 200)
    fig, ax = plt.subplots()
    ax.scatter(D, Y, s = 5, c = '#F0F0F0')
    ax.plot(x1, res.predict(x1), color = '#8DD2DB', label = 'RDD', linewidth = 2)
    ax.plot(x1, res.predict(x1) + res.left_ci - res.est, color = '#8DD2DB', linewidth = 1, alpha = 0.4)
    ax.plot(x1, res.predict(x1) + res.right_ci - res.est, color = '#8DD2DB', linewidth = 1, alpha = 0.4)
    ax.plot(x2, res.predict(x2), color = '#8DD2DB', linewidth = 2)
    ax.plot(x2, res.predict(x2) + res.left_ci - res.est, color = '#8DD2DB', linewidth = 1, alpha = 0.4)
    ax.plot(x2, res.predict(x2) + res.right_ci - res.est, color = '#8DD2DB', linewidth = 1, alpha = 0.4)
    ax.vlines(x = 0, ymin=-1.5, ymax=1.5, color='#000000', alpha = 0.7, linestyles='dashed')
    ax.legend(loc = 'upper left')
    ax.set_xlabel('D')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.8, 0.8)
    ax.spines[['right', 'top']].set_visible(False)
    return fig, ax

if __name__ == '__main__':
    main()