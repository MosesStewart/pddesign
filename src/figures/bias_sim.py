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

def main():
    outdir = 'output/figures'
    Y, W, D, Z, U = sim_biased(model_0, ndraws = 2000, seed = 1)
    
    model = rdd(Y, D, cutoff = 0.0, device = 'cuda', kernel = 'triangle', bandwidth = [0.50, 0.50])
    res_rdd = model.fit()
    print(res_rdd)
    
    model = pdd(Y, W, D, Z, cutoff = 0.0, device = 'cuda', kernel = 'triangle', bandwidth = [0.50, 0.50])
    res_pdd = model.fit()
    print(res_pdd)
    
    fig, ax = plot_res(res_rdd, res_pdd, Y, D, model_0)    
    fig.savefig(f'{outdir}/linear_example.pdf', transparent = True, bbox_inches="tight")

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
    ax.set_ylim(-1.35, 1.35)
    ax.spines[['right', 'top']].set_visible(False)
    return fig, ax

def plot_rdd(res, Y, D):
    x1 = np.linspace(-1.5, -0.001, 200)
    x2 = np.linspace(0, 1.5, 200)
    fig, ax = plt.subplots()
    ax.scatter(D, Y, s = 5, c = '#dddddd')
    ax.plot(x1, res.predict(x1), color = '#FF6961', label = 'RDD', linewidth = 2)
    ax.plot(x1, res.predict(x1) + res.left_ci - res.est, color = '#FF6961', linewidth = 1, alpha = 0.4)
    ax.plot(x1, res.predict(x1) + res.right_ci - res.est, color = '#FF6961', linewidth = 1, alpha = 0.4)
    ax.plot(x2, res.predict(x2), color = '#FF6961', linewidth = 2)
    ax.plot(x2, res.predict(x2) + res.left_ci - res.est, color = '#FF6961', linewidth = 1, alpha = 0.4)
    ax.plot(x2, res.predict(x2) + res.right_ci - res.est, color = '#FF6961', linewidth = 1, alpha = 0.4)
    ax.vlines(x = 0, ymin=-1.5, ymax=1.5, color='#000000', alpha = 0.7, linestyles='dashed')
    ax.legend(loc = 'upper left')
    ax.set_xlabel('D')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.5, 1.5)
    ax.spines[['right', 'top']].set_visible(False)
    return fig, ax

if __name__ == '__main__':
    main()
