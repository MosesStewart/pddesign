import numpy as np, pandas as pd, torch, sys, os, re
from matplotlib import pyplot as plt
sys.path.append('/'.join(re.split('/|\\\\', os.path.dirname( __file__ ))[0:-2]))
from rddesign.main import *

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
    df = pd.read_csv(f'{indir}/clean_data.csv')
    Y = df.loc[:, 'death'].values
    D = df.loc[:, 'brthwgt'].values
    W = df.loc[:, 'meduc'].values
    Z = df.loc[:, 'night'].values
    
    model = rdd(Y, D, cutoff = 1500, kernel = 'epan', bandwidth = [35, 35])
    res_rdd = model.fit()
    print(res_rdd)
    
    model = pdd(Y, W, D, Z, cutoff = 1500, kernel = 'epan', bandwidth = [35, 35])
    res_pdd = model.fit()
    print(res_pdd)
    
    fig, ax = plot_res(res_rdd, res_pdd, Y, D)    
    fig.savefig(f'{outdir}/results.pdf', transparent = True, bbox_inches="tight")
    
    model = rdd(W, D, cutoff = 1500, kernel = 'epan', bandwidth = [35, 35])
    res_rdd = model.fit()
    print(res_rdd)
    
    fig, ax = plot_rdd(res_rdd, W, D)
    ax.set_ylabel('$\\mathbb{E}\\left[W \\mid D = d\\right]$')
    fig.savefig(f'{outdir}/w_meduc.pdf', transparent = True, bbox_inches="tight")
    
    model = rdd(df.loc[:, 'medicaid'].values, D, cutoff = 1500, kernel = 'epan', bandwidth = [35, 35])
    res_rdd = model.fit()
    print(res_rdd)
    
    fig, ax = plot_rdd(res_rdd, df.loc[:, 'medicaid'].values, D)
    ax.set_ylabel('$\\mathbb{E}\\left[W \\mid D = d\\right]$')
    fig.savefig(f'{outdir}/w_medicaid.pdf', transparent = True, bbox_inches="tight")

def _add_side_brackets(ax, res, color, x_offset_neg=0.4, x_offset_pos=-0.4,
                       show_pos=True, show_neg=True,
                       pos_color=None, neg_color=None):
    cutoff = 1500
    cap = 0.08

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
        # Diagonal dotted lines from (cutoff, est) to bracket caps
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
        # Diagonal dotted lines from (cutoff, est) to bracket caps
        ax.plot([cutoff, xb], [est, lo],
                color=neg_color, linewidth=0.8, linestyle=':', alpha=0.7, zorder=4)
        ax.plot([cutoff, xb], [est, hi],
                color=neg_color, linewidth=0.8, linestyle=':', alpha=0.7, zorder=4)

def plot_res(rres, pres, Y, D):
    x1 = np.linspace(1490, 1499.99, 200)
    x2 = np.linspace(1500, 1510, 200)
    fig, ax = plt.subplots()
    ax.scatter(D, Y, s = 5, c = "#F0F0F0")
    ax.plot(x1, rres.predict(x1), color="#8DD2DB", label='RDD', linewidth=2)
    ax.plot(x2, rres.predict(x2), color='#8DD2DB', linewidth=2)
    ax.plot(x1, pres.predict(x1), color='#FF6961', label='PDD', linewidth=2)
    ax.plot(x2, pres.predict(x2) + 0.002, color='#FF6961', linewidth=2)
    # Right-side bracket: identical for RDD and PDD, draw once in black
    _add_side_brackets(ax, rres, color='#222222', x_offset_pos=-0.4, show_neg=False)
    # Left-side brackets: RDD further out, PDD closer in
    _add_side_brackets(ax, rres, color='#3aabb7', x_offset_neg=0.8, show_pos=False)
    _add_side_brackets(ax, pres, color='#d94f48', x_offset_neg=0.4, show_pos=False)
    ax.vlines(x = 1500, ymin=-0.1, ymax=1.05, color='#000000', alpha = 0.5, linestyle = (0, (5, 5)))
    ax.legend(loc='upper left')
    ax.set_xlabel('D')
    ax.set_ylabel('$\\mathbb{E}\\left[h_{0}(d, W) \\mid D = d \\right]$')
    ax.set_xlim(1490, 1510)
    ax.set_ylim(-0.1, 0.18)
    ax.spines[['right', 'top']].set_visible(False)
    return fig, ax

def plot_rdd(res, Y, D):
    x1 = np.linspace(1490, 1499.99, 200)
    x2 = np.linspace(1500, 1510, 200)
    fig, ax = plt.subplots()
    ax.scatter(D, Y, s=5, c='#dddddd')
    ax.plot(x1, res.predict(x1), color='#FF6961', label='RDD', linewidth=2)
    ax.plot(x2, res.predict(x2), color='#FF6961', linewidth=2)
    _add_side_brackets(ax, res, color='#d94f48', x_offset_neg=0.2, x_offset_pos=-0.2)
    ax.vlines(x = 1500, ymin=-0.1, ymax=1.05, color='#000000', alpha = 0.5, linestyle = (0, (5, 5)))
    ax.set_xlabel('D')
    ax.set_xlim(1490, 1510)
    ax.set_ylim(-0.05, 1.05)
    ax.spines[['right', 'top']].set_visible(False)
    return fig, ax

if __name__ == '__main__':
    main()