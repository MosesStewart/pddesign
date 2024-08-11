import numpy as np, pandas as pd
from main import RDD

def main():
    y, d, x = generate_data()
    x = pd.DataFrame(x, columns = ['First', 'Second'])
    model = RDD(y, d, exog = x, cutoff = 0)
    #res = model.fit_polynomial(method = 'aic')
    #res.summary()
    res = model.continuity_test(model = 'local linear')
    res.summary()
    #res = model.fit_polynomial(method = 'significance bins')
    #res.summary()
    #res = model.fit_local_lin()
    #res.summary()
    
def generate_data(seed = None):
    rng = np.random.default_rng(seed = seed)
    n = 400
    d = np.linspace(-10, 10, n)[:, None]
    a = np.where(d.flatten() > 0, 1, 0)[:, None]
    x = rng.multivariate_normal(mean = [1, 4], cov = np.array([[1, 0.5], [0.5, 1]]), size = n)#.squeeze(axis = 2)
    y = -1.5 * d + 1 * a  + rng.normal(scale = 1, size = n)[:, None] + x @ np.array([[-1], [1]])
    return y, d, x
    
if __name__ == '__main__':
    main()
