import numpy as np, pandas as pd, torch
from matplotlib import pyplot as plt
from main import *
from simulation import *

def main():
    Y, W, D, Z, U = linear_sim(ndraws = 2000, seed = 1001)
    model = pdd(Y, W, D, Z, cutoff = 0.0, device = 'cuda', kernel = 'triangle')
    res = model.fit()
    print(res)
    ds = np.linspace(-1, -0.01, 10)
    print(res.predict(ds))
    ds = np.linspace(0, 1, 10)
    print(res.predict(ds))

if __name__ == '__main__':
    main()
    