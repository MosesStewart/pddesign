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
    df = pd.read_csv(f'{indir}/clean_data_old.csv')
    Y = df.loc[:, 'death'].values
    D = df.loc[:, 'brthwgt'].values
    W = df.loc[:, 'meduc'].values
    Z = df.loc[:, 'wknd'].values
    
    model = rdd(Y, D, cutoff = 1500, kernel = 'rectangle', bandwidth = [85, 85])
    res_rdd = model.fit()
    print(res_rdd)
    
    model = pdd(Y, W, D, Z, cutoff = 1500, kernel = 'rectangle', bandwidth = [85, 85])
    res_pdd = model.fit()
    print(res_pdd)
    
    fig, ax = plot_res(res_rdd, res_pdd, Y, D)    
    fig.savefig(f'{outdir}/results_old.pdf', transparent = True, bbox_inches="tight")
    
    model = rdd(W, D, cutoff = 1500, kernel = 'rectangle', bandwidth = [85, 85])
    res_rdd = model.fit()
    print(res_rdd)
    
    fig, ax = plot_rdd(res_rdd, W, D)
    ax.set_ylabel('$\\mathbb{E}\\left[W \\mid D = d\\right]$')
    fig.savefig(f'{outdir}/w_meduc_old.pdf', transparent = True, bbox_inches="tight")
    
    model = rdd(df.loc[:, 'mrace'].values, D, cutoff = 1500, kernel = 'rectangle', bandwidth = [85, 85])
    res_rdd = model.fit()
    print(res_rdd)
    
    fig, ax = plot_rdd(res_rdd, df.loc[:, 'mrace'].values, D)
    ax.set_ylabel('$\\mathbb{E}\\left[W \\mid D = d\\right]$')
    fig.savefig(f'{outdir}/w_race_old.pdf', transparent = True, bbox_inches="tight")

if __name__ == '__main__':
    main()