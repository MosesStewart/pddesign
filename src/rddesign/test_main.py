import numpy as np, pandas as pd, torch
from matplotlib import pyplot as plt
from main import *
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
    outdir = 'temp'
    Y, W, D, Z, U = sim_bias(model_1, ndraws = 2000, seed = 3)
    fig, ax = dot_plot(D, U)
    ax.set_xlabel('D')
    ax.set_ylabel('U')
    fig.savefig(f'{outdir}/UD_scatter.png', bbox_inches='tight')
    
    model = rdd(Y, D, cutoff = 0.0, device = 'cuda', kernel = 'triangle', bandwidth = [0.50, 0.50])
    res_rdd = model.fit()
    print(res_rdd)
    
    model = pdd(Y, W, D, Z, cutoff = 0.0, device = 'cuda', kernel = 'triangle', bandwidth = [0.50, 0.50])
    res_pdd = model.fit()
    print(res_pdd)
    
    model = rdd(W, D, cutoff = 0.0, device = 'cuda', kernel = 'triangle', bandwidth = [0.50, 0.50])
    res_w = model.fit()
    fig, ax = plot_rdd(res_w, W, D)
    ax.set_ylabel('$\\mathbb{E}[W \\mid D = d]$')
    fig.savefig(f'{outdir}/W_rdd.png', bbox_inches = "tight")
    
    fig, ax = plot_res(res_rdd, res_pdd, Y, D, model_1)    
    fig.savefig(f'{outdir}/linear_dgp.png', bbox_inches="tight")
    
    '''
    successes = []
    margins = []
    for i in range(30):
        Y, W, D, Z, U = linear_sim(ndraws = 2000, seed = i)
        model = pdd(Y, W, D, Z, cutoff = 0.0, device = 'cuda', kernel = 'triangle')
        res = model.fit()
        if res.status == True:
            if -1 >= res.left_ci and -1 <= res.right_ci:
                successes.append(1)
            else:
                successes.append(0)
                margins.append(min(abs(res.left_ci + 1), abs(res.right_ci + 1)))
    print(np.mean(successes))
    print(margins)
    '''

if __name__ == '__main__':
    main()
    