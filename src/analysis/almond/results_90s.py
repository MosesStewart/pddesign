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
    
    with open(f'{outdir}/summary.txt', 'w') as summary:
        model = rdd(Y, D, cutoff = 1500, kernel = 'triangle', bandwidth = [85, 85])
        res_rdd = model.fit()
        summary.write('\nY ~ Death')
        summary.write(str(res_rdd))
        
        model = pdd(Y, W, D, Z, cutoff = 1500, kernel = 'triangle', bandwidth = [85, 85])
        res_pdd = model.fit()
        summary.write('\nY ~ Death')
        summary.write(str(res_pdd))
        
        fig, ax = plot_res(res_rdd, res_pdd, Y, D)
        ax.set_ylim(0, 0.08)
        fig.savefig(f'{outdir}/results_90s.pdf', transparent = True, bbox_inches="tight")
        
        model = rdd(W, D, cutoff = 1500, kernel = 'triangle', bandwidth = [85, 85])
        res_rdd = model.fit()
        summary.write('\nW ~ Meduc')
        summary.write(str(res_rdd))
        
        fig, ax = plot_rdd(res_rdd, W, D)
        ax.set_ylabel('$\\mathbb{E}\\left[W \\mid D = d\\right]$')
        ax.set_ylim(0.65, 0.85)
        fig.savefig(f'{outdir}/w_meduc_90s.pdf', transparent = True, bbox_inches="tight")
        
        model = rdd(df.loc[:, 'mrace'].values, D, cutoff = 1500, kernel = 'triangle', bandwidth = [85, 85])
        res_rdd = model.fit()
        summary.write('W ~ Mrace\n')
        summary.write(str(res_rdd))
        
        fig, ax = plot_rdd(res_rdd, df.loc[:, 'mrace'].values, D)
        ax.set_ylabel('$\\mathbb{E}\\left[W \\mid D = d\\right]$')
        ax.set_ylim(0.59, 0.80)
        fig.savefig(f'{outdir}/w_race_90s.pdf', transparent = True, bbox_inches="tight")

if __name__ == '__main__':
    main()