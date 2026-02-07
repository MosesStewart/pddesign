import torch, numpy as np, pandas as pd, sys, os, re
from matplotlib import pyplot as plt
sys.path.append('/'.join(re.split('/|\\\\', os.path.dirname( __file__ ))[0:-1]))
from derived.simulation import *

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE + 2)  # fontsize of the figure title
plt.rcParams["font.family"] = "Times New Roman"

def main():
    outdir = 'output/figures'
    models = {'Model 0 - Linear': model_0, 'Model 1 - Lee (2008)': model_1, 'Model 2 - Ludwig and Miller (2007)': model_2, 'Model 3 - Calonico et al. (2014)': model_3}
    
    fig, ax = plot_model('Model 0 - Linear', model_0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(-1.05, 1.05)
    fig.savefig(f'{outdir}/model_0.pdf', transparent = True, bbox_inches = 'tight')
    
    fig, ax = plot_model('Model 1 - Lee (2008)', model_1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-1.05, 1.05)
    fig.savefig(f'{outdir}/model_1.pdf', transparent = True, bbox_inches = 'tight')
    
    fig, ax = plot_model('Model 2 - Ludwig and Miller (2007)', model_2)
    ax.set_ylim(-0.05, 4.05)
    ax.set_xlim(-1.05, 1.05)
    fig.savefig(f'{outdir}/model_2.pdf', transparent = True, bbox_inches = 'tight')
    
    fig, ax = plot_model('Model 3 - Calonico et al. (2014)', model_3)
    ax.set_ylim(-5.05, 2.05)
    ax.set_xlim(-1.05, 1.05)
    fig.savefig(f'{outdir}/model_3.pdf', transparent = True, bbox_inches = 'tight')

def plot_model(name, dgp):
    fig, ax = plt.subplots()
    x_neg = np.linspace(-1.5, 0, 200)
    x_pos = np.linspace(0, 1.5, 200)
    ax.plot(x_neg[:-1], dgp(x_neg[:-1]), color = '#000000', label = 'Model')
    ax.plot(x_pos, dgp(x_pos), color = '#000000', label = 'Model')
    ax.set_title(name)
    ax.set_xlabel('d')
    ax.set_ylabel('$\\tilde{\\mu}_%s(d)$' % name[6])
    ax.vlines(x = 0, ymin=-6, ymax=6, color='#000000', alpha = 0.5, linestyle = (0, (5, 5)))
    ax.spines[['right', 'top']].set_visible(False)
    return fig, ax

if __name__ == '__main__':
    main()
