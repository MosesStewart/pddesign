import numpy as np, pandas as pd
from main import RDD

def main():
    y, d = generate_data()
    model = RDD(y, d, cutoff = 0)
    res = model.fit_local_lin()
    res.summary()
    
def generate_data(seed = None):
    rng = np.random.default_rng(seed = seed)
    n = 400
    d = np.linspace(-10, 10, n)[:, None]
    a = np.where(d.flatten() > 0, 1, 0)[:, None]
    y = -1 * d + 2 * a + rng.normal(scale = 1, size = n)[:, None]
    return y, d
    
if __name__ == '__main__':
    main()
