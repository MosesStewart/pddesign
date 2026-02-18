import numpy as np, pandas as pd, torch, sys, os, re
from matplotlib import pyplot as plt
sys.path.append('/'.join(re.split('/|\\\\', os.path.dirname( __file__ ))[0:-1]))
from rddesign.main import *
from derived.simulation import *

def main():
    outdir = 'temp'
    Y, W, D, Z, U = sim_unbiased(model_0, ndraws = 500, seed = 1)
    
    model = rdd(Y, D, cutoff = 0.0, device = 'cuda', kernel = 'triangle')
    res_rdd = model.fit()
    print(res_rdd)
    
    fig, ax = scatterplot(Y, D)
    ax.set_ylabel('Y')
    fig.savefig(f'{outdir}/YvsD.pdf', transparent = True, bbox_inches="tight")
    
    fig, ax = scatterplot(W, D)
    ax.set_ylabel('W')
    fig.savefig(f'{outdir}/WvsD.pdf', transparent = True, bbox_inches="tight")
    
    fig, ax = scatterplot(U, D)
    ax.set_ylabel('U')
    fig.savefig(f'{outdir}/UvsD.pdf', transparent = True, bbox_inches="tight")
    #model = pdd(Y, W, D, Z, cutoff = 0.0, device = 'cuda', kernel = 'triangle', bandwidth = [0.50, 0.50])
    #res_pdd = model.fit()
    
    #model = rdd(W, D, cutoff = 0.0, device = 'cuda', kernel = 'triangle', bandwidth = [0.50, 0.50])
    #res_w = model.fit()

def scatterplot(y, D, cutoff = 0):
    fig, ax = plt.subplots()
    ax.scatter(D, y, label = 'scatter', s = 5, c = '#FF6961')
    ax.vlines(x = cutoff, ymin=-1.5, ymax=1.5, color='#000000', alpha = 0.5, linestyle = (0, (5, 5)))
    ax.set_xlabel('D')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.35, 1.35)
    ax.spines[['right', 'top']].set_visible(False)
    return fig, ax

if __name__ == '__main__':
    main()
    