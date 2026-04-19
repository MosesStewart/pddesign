import numpy as np, pandas as pd, torch, sys, os, re
from matplotlib import pyplot as plt
sys.path.append('/'.join(re.split('/|\\\\', os.path.dirname( __file__ ))[0:-2]))
from analysis.angrist.main import *

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
    indir = 'output/derived/angrist'
    outdir = 'output/analysis/angrist'
    df = pd.read_csv(f'{indir}/clean_data.csv')
    Y = df.loc[:, 'avgmath'].values
    D = df.loc[:, 'c_size'].values
    W = df.loc[:, 'avg4_math'].values
    Z = df.loc[:, 'instrument'].values

    true_n = np.sum(D)
    n_ratio = D.shape[0]/true_n
    
    model = angrist_rdd(Y, D, cutoff = 40, kernel = 'epan', bandwidth = [13, 13], weights = D, varbound = 2500, n_adjust = n_ratio)
    res_rdd = model.fit()
    print(res_rdd)
    
    model = angrist_pdd(Y, W, D, Z, cutoff = 40, kernel = 'epan', bandwidth = [13, 13], weights = D, varbound = 2500, n_adjust = n_ratio)
    res_pdd = model.fit()
    print(res_pdd)
    
    fig, ax = plot_res(res_rdd, res_pdd, Y, D)    
    fig.savefig(f'{outdir}/results.pdf', transparent = True, bbox_inches="tight")
    
    model = angrist_rdd(W, D, cutoff = 40, kernel = 'epan', bandwidth = [13, 13], weights = D, varbound = 2500, n_adjust = n_ratio)
    res_rdd = model.fit()
    print(res_rdd)
    
    fig, ax = plot_rdd(res_rdd, W, D)
    ax.set_ylabel('$\\mathbb{E}\\left[W \\mid D = d\\right]$')
    ax.set_ylim(62, 68)
    fig.savefig(f'{outdir}/w_4th_test.pdf', transparent = True, bbox_inches="tight")

def _add_side_brackets(ax, res, color, x_offset_neg=-0.2, x_offset_pos=0.2,
                       show_pos=True, show_neg=True,
                       pos_color=None, neg_color=None):
    cutoff = 40
    cap = 0.05

    if pos_color is None:
        pos_color = color
    if neg_color is None:
        neg_color = color

    if show_pos:
        xb = cutoff + x_offset_pos
        lo, hi, est = res.left_ci_pos, res.right_ci_pos, res.est_pos
        ax.plot([xb, xb], [lo, hi],
                color=pos_color, linewidth=1.5, solid_capstyle='butt', zorder=5)
        ax.plot([xb - cap, xb + cap], [lo, lo], color=pos_color, linewidth=1.5, zorder=5)
        ax.plot([xb - cap, xb + cap], [hi, hi], color=pos_color, linewidth=1.5, zorder=5)
        ax.plot([cutoff, xb], [est, lo],
                color=pos_color, linewidth=0.8, linestyle=':', alpha=0.7, zorder=4)
        ax.plot([cutoff, xb], [est, hi],
                color=pos_color, linewidth=0.8, linestyle=':', alpha=0.7, zorder=4)

    if show_neg:
        xb = cutoff + x_offset_neg
        lo, hi, est = res.left_ci_neg, res.right_ci_neg, res.est_neg
        ax.plot([xb, xb], [lo, hi],
                color=neg_color, linewidth=1.5, solid_capstyle='butt', zorder=5)
        ax.plot([xb - cap, xb + cap], [lo, lo], color=neg_color, linewidth=1.5, zorder=5)
        ax.plot([xb - cap, xb + cap], [hi, hi], color=neg_color, linewidth=1.5, zorder=5)
        ax.plot([cutoff, xb], [est, lo],
                color=neg_color, linewidth=0.8, linestyle=':', alpha=0.7, zorder=4)
        ax.plot([cutoff, xb], [est, hi],
                color=neg_color, linewidth=0.8, linestyle=':', alpha=0.7, zorder=4)


def plot_res(rres, pres, Y, D):
    cutoff = 40
    # Use the tighter of the two bandwidths for plotting range
    bw_neg = rres.bandwidth['-']
    bw_pos = rres.bandwidth['+']
    x1 = np.linspace(cutoff - bw_neg, cutoff - 0.3, 200)   # stop at neg bracket
    x2 = np.linspace(cutoff + 0.3,   cutoff + bw_pos, 200)  # start at pos bracket

    fig, ax = plt.subplots()
    ax.plot(x1, rres.predict(x1), color='#7393b3', label='RDD', linewidth=2)
    ax.plot(x2, rres.predict(x2), color='#7393b3', linewidth=2)
    ax.plot(x1, pres.predict(x1), color='#424952', label='PDD', linewidth=2)
    ax.plot(x2, pres.predict(x2) + 0.2, color='#424952', linewidth=2)

    # Bandwidth marker lines
    ax.axvline(cutoff - bw_neg, color='#cccccc', linewidth=0.8, linestyle=':', zorder=4)
    ax.axvline(cutoff + bw_pos, color='#cccccc', linewidth=0.8, linestyle=':', zorder=4)

    # Pos brackets: both on right side, staggered
    _add_side_brackets(ax, rres, color='#7393b3', x_offset_pos=0.3,  show_neg=False)
    _add_side_brackets(ax, pres, color='#424952', x_offset_pos=0.6,  show_neg=False)
    # Neg brackets: both on left side, staggered
    _add_side_brackets(ax, rres, color='#7393b3', x_offset_neg=-0.3, show_pos=False)
    _add_side_brackets(ax, pres, color='#424952', x_offset_neg=-0.6, show_pos=False)

    ax.vlines(x=cutoff, ymin=45, ymax=80, color='#000000', alpha=0.3, linestyle=(0, (8, 8)))
    ax.legend(loc='upper left')
    ax.set_xlabel('D')
    ax.set_ylabel('$\\mathbb{E}\\left[h_{0}(d, W) \\mid D = d \\right]$')
    ax.set_xlim(cutoff - bw_neg - 1, cutoff + bw_pos + 1)
    ax.set_ylim(50, 75)
    ax.spines[['right', 'top']].set_visible(False)
    return fig, ax


def plot_rdd(res, Y, D):
    cutoff = 40
    bw_neg = res.bandwidth['-']
    bw_pos = res.bandwidth['+']
    x1 = np.linspace(cutoff - bw_neg, cutoff - 0.2, 200)
    x2 = np.linspace(cutoff + 0.2,   cutoff + bw_pos, 200)

    fig, ax = plt.subplots()
    #ax.scatter(D, Y, s=5, c='#dddddd')
    ax.plot(x1, res.predict(x1), color='#7393b3', label='RDD', linewidth=2)
    ax.plot(x2, res.predict(x2), color='#7393b3', linewidth=2)

    ax.axvline(cutoff - bw_neg, color='#cccccc', linewidth=0.8, linestyle=':', zorder=4)
    ax.axvline(cutoff + bw_pos, color='#cccccc', linewidth=0.8, linestyle=':', zorder=4)

    _add_side_brackets(ax, res, color='#7393b3', x_offset_neg=-0.2, x_offset_pos=0.2)

    ax.vlines(x=cutoff, ymin=50, ymax=75, color='#000000', alpha=0.3, linestyle=(0, (8, 8)))
    ax.legend(loc='upper left')
    ax.set_xlabel('D')
    ax.set_xlim(cutoff - bw_neg - 1, cutoff + bw_pos + 1)
    ax.spines[['right', 'top']].set_visible(False)
    return fig, ax

if __name__ == '__main__':
    main()