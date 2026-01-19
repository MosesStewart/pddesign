import numpy as np, pandas as pd, torch
from matplotlib import pyplot as plt
from main import *
from simulation import *

def main():
    '''
    Y, W, D, Z, U = linear_sim(ndraws = 2000, seed = 2)
    model = pdd(Y, W, D, Z, cutoff = 0.0, device = 'cuda', kernel = 'gaussian')
    res = model.fit()
    print(res)
    ds = np.linspace(-1, -0.001, 10)
    print(res.predict(ds))
    ds = np.linspace(0, 1, 10)
    print(res.predict(ds))
    
    '''
    successes = []
    margins = []
    for i in range(30):
        Y, W, D, Z, U = linear_sim(ndraws = 2000, seed = i)
        model = pdd(Y, W, D, Z, cutoff = 0.0, device = 'cuda', kernel = 'triangle')
        res = model.fit()
        if res.status == True:
            if -1 >= res.left_ci and -1 <= res.right_ci:
                successes.append(1)
            else:
                successes.append(0)
                margins.append(min(abs(res.left_ci + 1), abs(res.right_ci + 1)))
    print(np.mean(successes))
    print(margins)

if __name__ == '__main__':
    main()
    