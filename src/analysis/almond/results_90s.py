import numpy as np, pandas as pd, torch, sys, os, re
from matplotlib import pyplot as plt
sys.path.append('/'.join(re.split('/|\\\\', os.path.dirname( __file__ ))[0:-2]))
from rddesign.main import *
from analysis.almond.results import _add_side_brackets, plot_rdd
from analysis.almond.results_90s_nobc import rdd_nobc

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
        
        model = rdd_nobc(Y, D, cutoff = 1500, kernel = 'uniform', bandwidth = [85, 85])
        res_rdd_nobc = model.fit()
        summary.write('\nY ~ Death -- No bias correction')
        summary.write(str(res_rdd_nobc))
        
        fig, ax = plot_res(res_rdd, res_pdd, res_rdd_nobc, Y, D)
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


def plot_res(rres, pres, nobcres, Y, D):
    cutoff = 1500
    # Use the tighter of the two bandwidths for plotting range
    bw_neg = rres.bandwidth['-']
    bw_pos = rres.bandwidth['+']
    x1 = np.linspace(cutoff - bw_neg, cutoff - 2.5, 200)   # stop at neg bracket
    x2 = np.linspace(cutoff + 2.5,   cutoff + bw_pos, 200)  # start at pos bracket

    fig, ax = plt.subplots()
    ax.scatter(D, Y, s=5, c='#eeeeee')
    ax.plot(x1, rres.predict(x1), color='#7393b3', label='RDD', linewidth=2)
    ax.plot(x2, rres.predict(x2), color='#7393b3', linewidth=2)
    ax.plot(x1, pres.predict(x1), color='#424952', label='PDD', linewidth=2)
    ax.plot(x2, pres.predict(x2) + 0.0005, color='#424952', linewidth=2)
    ax.plot(x1, nobcres.predict(x1), color="#8E96A1", label='OG', linewidth=2)
    ax.plot(x2, nobcres.predict(x2), color='#8E96A1', linewidth=2)
    
    # Bandwidth marker lines
    ax.axvline(cutoff - bw_neg, color='#bbbbbb', linewidth=0.8, linestyle=':', zorder=4)
    ax.axvline(cutoff + bw_pos, color='#bbbbbb', linewidth=0.8, linestyle=':', zorder=4)

    # Pos brackets: both on right side, staggered
    _add_side_brackets(ax, rres, color='#7393b3', x_offset_pos=2.5,  show_neg=False)
    _add_side_brackets(ax, pres, color='#424952', x_offset_pos=5,  show_neg=False)
    _add_side_brackets(ax, nobcres, color='#8E96A1', x_offset_pos=3.75,  show_neg=False)
    # Neg brackets: both on left side, staggered
    _add_side_brackets(ax, rres, color='#7393b3', x_offset_neg=-2.5, show_pos=False)
    _add_side_brackets(ax, pres, color='#424952', x_offset_neg=-5, show_pos=False)
    _add_side_brackets(ax, nobcres, color='#8E96A1', x_offset_neg=-3.75,  show_pos=False)

    ax.vlines(x = 1500, ymin=-0.1, ymax=1.05, color='#000000', alpha=0.3, linestyle=(0, (8, 8)))
    ax.legend(loc='upper left')
    ax.set_xlabel('D')
    ax.set_ylabel('$\\mathbb{E}\\left[h_{0}(d, W) \\mid D = d \\right]$')
    ax.set_xlim(cutoff - bw_neg - 1, cutoff + bw_pos + 1)
    ax.set_ylim(-0.1, 0.18)
    ax.spines[['right', 'top']].set_visible(False)
    return fig, ax

if __name__ == '__main__':
    main()