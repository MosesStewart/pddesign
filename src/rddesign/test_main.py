import numpy as np, pandas as pd
from main import RDD

def main():
    y, d, x, a = generate_data()
    xgrid = np.linspace(-2, 2, 20)
    model = RDD(y, d, treatment = a, exog = x, cutoff = 0)
    res = model.fit(model = 'polynomial', design = 'fuzzy', nreps = 100)
    yhat = res.predict(xgrid)
    print(res)
    
def generate_data(seed = None):
    rng = np.random.default_rng(seed = seed)
    n = 400
    d = np.linspace(-10, 10, n)[:, None]
    Pa = np.where(d.flatten() > 0, 0.9, 0.1)[:, None]
    a = rng.binomial(1, Pa)
    x = rng.multivariate_normal(mean = [1, 4], cov = np.array([[1, 0.5], [0.5, 1]]), size = n)#.squeeze(axis = 2)
    y = -1.5 * d + 2 * a  + rng.normal(scale = 1, size = n)[:, None] + x @ np.array([[-1], [1]])
    return y, d, x, a
    
if __name__ == '__main__':
    main()
