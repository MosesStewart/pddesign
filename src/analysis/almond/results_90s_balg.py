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
    n, m, reps = Y.shape[0], 1000, 5
    h_pos, h_neg = [], []
    rng = np.random.default_rng(seed = 10042002)
    
    with open(f'{outdir}/summary_balg.txt', 'w') as summary:
        
        for rep in range(reps):
            bsample = rng.choice(n, size = m, replace = False)
            model = rdd(Y[bsample], D[bsample], cutoff = 1500, kernel = 'triangle')
            res = model.fit()
            h_neg.append((m**(-1/5)/n**(-1/5)) * res.bandwidth['-'])
            h_pos.append((m**(-1/5)/n**(-1/5)) * res.bandwidth['+'])
            
        h = {'-': np.mean(h_neg), '+': np.mean(h_pos)}
        
        model = rdd(Y, D, cutoff = 1500, kernel = 'triangle', bandwidth = [h['-'], h['+']], dtype = torch.float32)
        res_rdd = model.fit()
        summary.write('\nY ~ Death')
        summary.write(str(res_rdd))
        
        for rep in range(reps):
            bsample = rng.choice(n, size = m, replace = False)
            model = pdd(Y[bsample], W[bsample], D[bsample], Z[bsample], cutoff = 1500, kernel = 'triangle')
            res = model.fit()
            h_neg.append((m**(-1/5)/n**(-1/5)) * res.bandwidth['-'])
            h_pos.append((m**(-1/5)/n**(-1/5)) * res.bandwidth['+'])
            
        h = {'-': np.mean(h_neg), '+': np.mean(h_pos)}
        
        model = pdd(Y, W, D, Z, cutoff = 1500, kernel = 'triangle', bandwidth = [h['-'], h['+']], dtype = torch.float32)
        res_pdd = model.fit()
        summary.write('\nY ~ Death')
        summary.write(str(res_pdd))
        
        fig, ax = plot_res(res_rdd, res_pdd, Y, D)
        ax.set_ylim(0, 0.08)
        fig.savefig(f'{outdir}/results_90s_balg.pdf', transparent = True, bbox_inches="tight")

if __name__ == '__main__':
    main()