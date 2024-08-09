import numpy as np, pandas as pd
from main import RDD

def main():
    y, d = generate_data()
    model = RDD(y, d, cutoff = 0)
    res = model.fit_polynomial(method = 'aic')
    res.summary()
    print('\n')
    res = model.fit_polynomial(method = 'aic')
    res.summary()
    #res = model.fit_local_lin()
    #res.summary()
    
def generate_data(seed = None):
    rng = np.random.default_rng(seed = seed)
    n = 400
    d = np.linspace(-10, 10, n)[:, None]
    a = np.where(d.flatten() > 0, 1, 0)[:, None]
    y = -3 * d + 2 * a + 1 * d**2 + rng.normal(scale = 2, size = n)[:, None]
    return y, d
    
if __name__ == '__main__':
    main()
