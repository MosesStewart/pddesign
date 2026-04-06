import numpy as np, pandas as pd, torch, sys, os, re
from matplotlib import pyplot as plt
sys.path.append('/'.join(re.split('/|\\\\', os.path.dirname( __file__ ))[0:-2]))
from rddesign.main import *
from derived.simulation import *

def main():
    indir = 'output/derived/angrist'
    outdir = 'output/analysis/angrist'
    df = pd.read_csv(f'{indir}/clean_data.csv')
    Y = df.loc[:, 'avgmath'].values
    D = df.loc[:, 'c_size'].values
    W = df.loc[:, 'avg4_math'].values
    Z = df.loc[:, 'instrument'].values

    model = rdd(Y, D, cutoff = 40, kernel = 'epan', bandwidth = [20, 20])
    res_rdd = model.fit()
    print(res_rdd)
    
    model = pdd(Y, W, D, Z, cutoff = 40, kernel = 'epan', bandwidth = [20, 20])
    res_pdd = model.fit()
    print(res_pdd)
    
    fig, ax = plot_res(res_rdd, res_pdd, Y, D)    
    fig.savefig(f'{outdir}/results.pdf', transparent = True, bbox_inches="tight")
    
    model = rdd(W, D, cutoff = 40, kernel = 'epan', bandwidth = [20, 20])
    res_rdd = model.fit()
    print(res_rdd)
    
    fig, ax = plot_rdd(res_rdd, W, D)
    ax.set_ylabel('$\\mathbb{E}\\left[W \\mid D = d\\right]$')
    ax.set_ylim(58, 75)
    fig.savefig(f'{outdir}/w_4th_test.pdf', transparent = True, bbox_inches="tight")

def plot_res(rres, pres, Y, D):
    x1 = np.linspace(35, 39.99, 200)
    x2 = np.linspace(40, 45, 200)
    fig, ax = plt.subplots()
    ax.scatter(D, Y, s = 5, c = "#F0F0F0")
    ax.plot(x1, rres.predict(x1), color = "#8DD2DB", label = 'RDD', linewidth = 2)
    ax.plot(x1, rres.predict(x1) + (rres.left_ci - rres.est)/2, color = "#8DD2DB", linewidth = 1, alpha = 0.55)
    ax.plot(x1, rres.predict(x1) + (rres.right_ci - rres.est)/2, color = '#8DD2DB', linewidth = 1, alpha = 0.55)
    ax.plot(x1, pres.predict(x1), color = '#FF6961', label = 'PDD', linewidth = 2)
    ax.plot(x1, pres.predict(x1) + (pres.left_ci - pres.est)/2, color = '#FF6961', linewidth = 1, alpha = 0.55)
    ax.plot(x1, pres.predict(x1) + (pres.right_ci - pres.est)/2, color = '#FF6961', linewidth = 1, alpha = 0.55)
    ax.plot(x2, rres.predict(x2), color = '#8DD2DB', linewidth = 2)
    ax.plot(x2, rres.predict(x2) + (rres.left_ci - rres.est)/2, color = '#8DD2DB', linewidth = 1, alpha = 0.55)
    ax.plot(x2, rres.predict(x2) + (rres.right_ci - rres.est)/2, color = '#8DD2DB', linewidth = 1, alpha = 0.55)
    ax.plot(x2, pres.predict(x2), color = '#FF6961', linewidth = 2)
    ax.plot(x2, pres.predict(x2) + (pres.left_ci - pres.est)/2, color = '#FF6961', linewidth = 1, alpha = 0.55)
    ax.plot(x2, pres.predict(x2) + (pres.right_ci - pres.est)/2, color = '#FF6961', linewidth = 1, alpha = 0.55)
    ax.vlines(x = 1500, ymin=-0.1, ymax=1.05, color='#000000', alpha = 0.5, linestyle = (0, (5, 5)))
    ax.legend(loc = 'upper left')
    ax.set_xlabel('D')
    ax.set_ylabel('$\\mathbb{E}\\left[h_{0}(d, W) \\mid D = d \\right]$')
    ax.set_xlim(35, 45)
    ax.set_ylim(55, 75)
    ax.spines[['right', 'top']].set_visible(False)
    return fig, ax

def plot_rdd(res, Y, D):
    x1 = np.linspace(35, 39.99, 200)
    x2 = np.linspace(40, 45, 200)
    fig, ax = plt.subplots()
    ax.scatter(D, Y, s = 5, c = '#dddddd')
    ax.plot(x1, res.predict(x1), color = '#FF6961', label = 'RDD', linewidth = 2)
    ax.plot(x1, res.predict(x1) + (res.left_ci - res.est)/2, color = '#FF6961', linewidth = 1, alpha = 0.4)
    ax.plot(x1, res.predict(x1) + (res.right_ci - res.est)/2, color = '#FF6961', linewidth = 1, alpha = 0.4)
    ax.plot(x2, res.predict(x2), color = '#FF6961', linewidth = 2)
    ax.plot(x2, res.predict(x2) + (res.left_ci - res.est)/2, color = '#FF6961', linewidth = 1, alpha = 0.4)
    ax.plot(x2, res.predict(x2) + (res.right_ci - res.est)/2, color = '#FF6961', linewidth = 1, alpha = 0.4)
    ax.vlines(x = 1500, ymin=-0.1, ymax=1.05, color='#000000', alpha = 0.7, linestyles='dashed')
    ax.legend(loc = 'upper left')
    ax.set_xlabel('D')
    ax.set_xlim(35, 45)
    ax.spines[['right', 'top']].set_visible(False)
    return fig, ax

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