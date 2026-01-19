import numpy as np, pandas as pd, torch
from matplotlib import pyplot as plt
from main import *
from simulation import *

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
    Y, W, D, Z, U = linear_sim(ndraws = 2000, seed = 14)
    dgp = lambda x: (x >= 0) * 1.5 + (x < 0) * 0.5 + x/2
    model = rdd(Y, D, cutoff = 0.0, device = 'cuda', kernel = 'triangle', bandwidth = [0.50, 0.50])
    res_rdd = model.fit()
    print(res_rdd)
    
    model = pdd(Y, W, D, Z, cutoff = 0.0, device = 'cuda', kernel = 'triangle', bandwidth = [0.50, 0.50])
    res_pdd = model.fit()
    print(res_pdd)
    
    fig, ax = plot_res(res_rdd, res_pdd, Y, D, dgp)    
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

def plot_res(rres, pres, Y, D, fn):
    x1 = np.linspace(-1.5, -0.001, 200)
    x2 = np.linspace(0, 1.5, 200)
    fig, ax = plt.subplots()
    ax.scatter(D, Y, s = 5, c = '#dddddd')
    ax.plot(x1, fn(x1), color = '#80EF80', label = 'True', linewidth = 2)
    ax.plot(x1, rres.predict(x1), color = '#B3EBF2', label = 'RDD', linewidth = 2)
    ax.plot(x1, rres.predict(x1) + rres.left_ci - rres.est, color = '#B3EBF2', linewidth = 1, alpha = 0.2)
    ax.plot(x1, rres.predict(x1) + rres.right_ci - rres.est, color = '#B3EBF2', linewidth = 1, alpha = 0.2)
    ax.plot(x1, pres.predict(x1), color = '#FF6961', label = 'PDD', linewidth = 2)
    ax.plot(x1, pres.predict(x1) + pres.left_ci - pres.est, color = '#FF6961', linewidth = 1, alpha = 0.2)
    ax.plot(x1, pres.predict(x1) + pres.right_ci - pres.est, color = '#FF6961', linewidth = 1, alpha = 0.2)
    ax.plot(x2, fn(x2), color = '#80EF80', label = 'True', linewidth = 2)
    ax.plot(x2, rres.predict(x2), color = '#B3EBF2', linewidth = 2)
    ax.plot(x2, rres.predict(x2) + rres.left_ci - rres.est, color = '#B3EBF2', linewidth = 1, alpha = 0.2)
    ax.plot(x2, rres.predict(x2) + rres.right_ci - rres.est, color = '#B3EBF2', linewidth = 1, alpha = 0.2)
    ax.plot(x2, pres.predict(x2), color = '#FF6961', linewidth = 2)
    ax.plot(x2, pres.predict(x2) + pres.left_ci - pres.est, color = '#FF6961', linewidth = 1, alpha = 0.2)
    ax.plot(x2, pres.predict(x2) + pres.right_ci - pres.est, color = '#FF6961', linewidth = 1, alpha = 0.2)
    ax.legend(loc = 'upper left')
    ax.set_xlabel('D')
    ax.set_ylabel('$\\mathbb{E}[Y \\mid D = d]$')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-1.5, 2.5)
    ax.spines[['right', 'top']].set_visible(False)
    return fig, ax

if __name__ == '__main__':
    main()
    