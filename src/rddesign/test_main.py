import numpy as np, pandas as pd
from main import RDD

def main():
    y, d, x, a = generate_sample_data(seed = 1004)
    xgrid = np.linspace(-2, 2, 20)
    model = RDD(y, d, treatment = a, exog = x, cutoff = 0)
    res = model.fit(model = "polynomial", design = "fuzzy", optimization = 'aic', bootstrap = True,
              nreps = 1000, seed = 2002)
    print(res)
    
def generate_sample_data(seed = None):
    rng = np.random.default_rng(seed = seed)
    n = 400
    d = np.linspace(-10, 10, n)[:, None]
    Pa = np.where(d.flatten() > 0, 0.9, 0.1)[:, None]
    a = rng.binomial(1, Pa)
    x = rng.multivariate_normal(mean = [1, 4], cov = np.array([[0.5, 0.25], [0.25, 0.5]]), size = n)
    y = -1.5 * d + 0.05 * d**2 + 2 * a + x @ np.array([[-1], [1]]) + rng.normal(scale = 1, size = n)[:, None]
    return y, d, x, a
    
if __name__ == '__main__':
    main()
