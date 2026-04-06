import numpy as np, pandas as pd, torch, sys, os, re
from matplotlib import pyplot as plt
sys.path.append('/'.join(re.split('/|\\\\', os.path.dirname( __file__ ))[0:-2]))
from rddesign.main import *
from derived.simulation import *

def main():
    pass

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