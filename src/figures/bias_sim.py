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

def _add_side_brackets(ax, res, color, x_offset_neg=-0.2, x_offset_pos=0.2,
                       show_pos=True, show_neg=True, pos_color=None, neg_color=None, cutoff=0):
    cap = 0.01

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


def plot_res(rres, pres, Y, D, fn):
    cutoff = 0
    # Use the tighter of the two bandwidths for plotting range
    bw_neg = rres.bandwidth['-']
    bw_pos = rres.bandwidth['+']
    x1 = np.linspace(-1, cutoff - 0.03, 200)   # stop at neg bracket
    x2 = np.linspace(cutoff + 0.03, 1, 200)  # start at pos bracket

    fig, ax = plt.subplots()
    ax.scatter(D, Y, s=5, c='#eeeeee')
    ax.plot(x1, fn(x1), color = "#1a9c6f", label = 'True', linewidth = 2)
    ax.plot(x2, fn(x2), color = "#1a9c6f", linewidth = 2)
    ax.plot(x1, rres.predict(x1), color='#7393b3', label='RDD', linewidth=2)
    ax.plot(x2, rres.predict(x2), color='#7393b3', linewidth=2)
    ax.plot(x1, pres.predict(x1), color='#424952', label='PDD', linewidth=2)
    ax.plot(x2, pres.predict(x2) + 0.02, color='#424952', linewidth=2)

    # Bandwidth marker lines
    ax.axvline(cutoff - bw_neg, color='#bbbbbb', linewidth=0.8, linestyle=':', zorder=4)
    ax.axvline(cutoff + bw_pos, color='#bbbbbb', linewidth=0.8, linestyle=':', zorder=4)

    # Pos brackets: both on right side, staggered
    _add_side_brackets(ax, rres, color='#7393b3', x_offset_pos=0.03,  show_neg=False)
    _add_side_brackets(ax, pres, color='#424952', x_offset_pos=0.06,  show_neg=False)
    # Neg brackets: both on left side, staggered
    _add_side_brackets(ax, rres, color='#7393b3', x_offset_neg=-0.03, show_pos=False)
    _add_side_brackets(ax, pres, color='#424952', x_offset_neg=-0.06, show_pos=False)

    ax.vlines(x=cutoff, ymin=-1.5, ymax=1.5, color='#000000', alpha=0.3, linestyle=(0, (8, 8)))
    ax.legend(loc='upper left')
    ax.set_xlabel('D')
    ax.set_ylabel('$\\tilde{\\mu}_{0}(D)$')
    ax.set_xlim(cutoff - bw_neg - 1, cutoff + bw_pos + 1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.35, 1.35)
    ax.spines[['right', 'top']].set_visible(False)
    return fig, ax


def plot_rdd(res, Y, D):
    cutoff = 0
    bw_neg = res.bandwidth['-']
    bw_pos = res.bandwidth['+']
    x1 = np.linspace(cutoff - bw_neg, cutoff - 0.4, 200)
    x2 = np.linspace(cutoff + 0.4,   cutoff + bw_pos, 200)

    fig, ax = plt.subplots()
    ax.scatter(D, Y, s=5, c='#eeeeee')
    ax.plot(x1, res.predict(x1), color='#7393b3', label='RDD', linewidth=2)
    ax.plot(x2, res.predict(x2), color='#7393b3', linewidth=2)

    ax.axvline(cutoff - bw_neg, color='#bbbbbb', linewidth=0.8, linestyle=':', zorder=4)
    ax.axvline(cutoff + bw_pos, color='#bbbbbb', linewidth=0.8, linestyle=':', zorder=4)

    _add_side_brackets(ax, res, color='#7393b3', x_offset_neg=-0.4, x_offset_pos=0.4)

    ax.vlines(x=cutoff, ymin=-1.5, ymax=1.5, color='#000000', alpha=0.3, linestyle=(0, (8, 8)))
    ax.legend(loc='upper left')
    ax.set_xlabel('D')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.35, 1.35)
    ax.spines[['right', 'top']].set_visible(False)
    return fig, ax

if __name__ == '__main__':
    main()
