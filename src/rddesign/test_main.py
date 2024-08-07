import numpy as np, pandas as pd
from main import RDD

def main():
    y, d = generate_data()
    model = RDD(y, d, cutoff = 0)
    
def generate_data(seed = 10042002):
    rng = np.random.default_rng(seed)
    n = 200
    d = np.linspace(-10, 10, n)[:, None]
    a = np.where(d.flatten() > 0, 1, 0)[:, None]
    y = -1 * d + 2 * a + rng.normal(scale = 2, size = n)[:, None]
    return y, d
    
if __name__ == '__main__':
    main()
