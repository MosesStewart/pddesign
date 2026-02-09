import numpy as np, pandas as pd, torch, sys, os, re
from matplotlib import pyplot as plt
sys.path.append('/'.join(re.split('/|\\\\', os.path.dirname( __file__ ))[0:-1]))
from rddesign.main import *
from derived.simulation import *

def main():
    outdir = 'temp'
    Y, W, D, Z, U = sim_unbiased(model_0, ndraws = 500, seed = 3)

    #model = rdd(Y, D, cutoff = 0.0, device = 'cuda', kernel = 'triangle', bandwidth = [0.50, 0.50])
    #res_rdd = model.fit()
    #print(res_rdd)
    
    model = pdd(Y, W, D, Z, cutoff = 0.0, device = 'cuda', kernel = 'triangle', bandwidth = [0.50, 0.50])
    res_pdd = model.fit()
    print(res_pdd)
    print(res_pdd.se)
    
    #model = rdd(W, D, cutoff = 0.0, device = 'cuda', kernel = 'triangle', bandwidth = [0.50, 0.50])
    #res_w = model.fit()

if __name__ == '__main__':
    main()
    