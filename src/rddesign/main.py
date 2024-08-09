import numpy as np, pandas as pd, warnings
from scipy.optimize import minimize
from scipy.stats import t

class RDD():
    
    def __init__(self, outcome, runv, cutoff, treatmnt = None, exog = None, weights = None, **kwargs):
        self.y_name = None
        self.d_name = None
        self.x_names = None
        self.a_names = None
        self.cutoff = cutoff
        
        if type(outcome) == pd.DataFrame:
            self.y = outcome.to_numpy()
            self.y_name = outcome.columns[0]
        elif type(outcome) == np.ndarray:
            self.y = np.reshape(outcome, (outcome.shape[0], 1))
            
        self.n = self.y.shape[0]
        
        if type(runv) == pd.DataFrame:
            self.d = runv.to_numpy()
            self.d_name = runv.columns[0]
        elif type(runv) == np.ndarray:
            self.d = np.reshape(runv, (self.n, 1))
            
        if type(exog) == pd.DataFrame:
            self.x = exog.to_numpy()
            self.x_names = exog.columns
        elif type(exog) == np.ndarray:
            self.x = np.reshape(exog, (self.n, 1))
        else:
            self.x = None
            
        if type(treatmnt) == pd.DataFrame:
            self.a = treatmnt.to_numpy()
            self.a_names = treatmnt.columns
        elif type(exog) == np.ndarray:
            self.a = np.reshape(treatmnt, (self.n, 1))
        else:
            self.a = np.where(self.d.flatten() > cutoff, 1, 0)
        
        if type(weights) == pd.DataFrame:
            wgts = weights.to_numpy()
            self.weights = wgts/np.mean(wgts)
        elif type(weights) == np.ndarray:
            self.weights = weights.flatten()/np.mean(weights.flatten())
        else:
            self.weights = np.ones(self.n)
        
        
    def fit_local_lin(self, bandwidth = None):
        '''
        Fits the model using a local linear regression
        
        :param bandwidth: the width of the window to the left and
            right of the cutoff to use for the local regression
        '''
        if bandwidth == None:
            self._find_bandwidth()
        else:
            self.bandwidth = bandwidth
        
        use = np.where((self.cutoff - self.bandwidth <= self.d) & (self.cutoff + self.bandwidth >= self.d))[0]
        n = use.shape[0]
        
        d = self.d[use,:] - self.cutoff
        isright = np.where(d >= self.cutoff, 1, 0)
        slope_adj = isright * d
        X = np.concatenate([isright, np.ones((d.shape[0], 1)), d, slope_adj], axis = 1)
        W = np.diag(self.weights[use])
        Y = self.y[use,:]
        
        coefs = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y
        
        r = Y - X @ coefs
        M = np.diag(r.flatten()**2)
        varcov = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ M @ W.T @ X @ np.linalg.inv(X.T @ W @ X)
        tstat = np.sqrt(n - 3) * coefs[0, 0]/np.sqrt(varcov[0, 0])
        pval = min(t.cdf(tstat, n - 3) + (1 - t.cdf(-tstat, n - 3)), t.cdf(-tstat, n - 3) + (1 - t.cdf(tstat, n - 3)))
        
        results = Results()
        results.params = pd.Series({'Treatment': coefs[0, 0]}, name = 'Estimated effect')
        results.bse = pd.Series({'Treatment': np.sqrt(varcov[0, 0])}, name = 'Standard error')
        results.resid = r.flatten()
        results.tvalues = pd.Series({'Treatment': tstat}, name = 't-statistic')
        results.pvalues = pd.Series({'Treatment': pval}, name = 'p-value != 0')
        results.bandwidth = self.bandwidth
        results.model = 'Sharp Local Linear Regression'
        results.d_name = self.d_name
        results.y_name = self.y_name
        results.n = n
        return results

    def _find_bandwidth(self):
        '''
        Uses the leave-one-out procedure from Jens Ludwig and
            Douglas Miller (2007) and Imbens and Lemieux (2008)
            to compute optimal bandwidth for local linear regression
        '''
        d_sorted = np.sort(self.d)
        h0 = 10 * np.mean(d_sorted[1:] - d_sorted[:-1])
        
        def bandwidth_loss(h):
            use_left = np.where((self.d.flatten() < self.cutoff) & (self.d.flatten() >= np.min(self.d) + h))[0]
            use_right = np.where((self.d.flatten() >= self.cutoff) & (self.d.flatten() <= np.max(self.d) - h))[0]
            if use_left.shape[0] == 0 or use_right.shape[0] == 0:
                return np.inf                               # make sure we aren't usinng the whole side
            else:
                left_grid, right_grid = self.d[use_left, :], self.d[use_right, :]
                left_est, right_est = np.zeros((left_grid.shape[0], 1)), np.zeros((right_grid.shape[0], 1))
                left_weights, right_weights = self.weights[use_left][:, None]/self.n, self.weights[use_right][:, None]/self.n
                left_true, right_true = self.y[use_left, :], self.y[use_right, :]
                
                for i in range(left_grid.shape[0]):
                    d = left_grid[i, 0]
                    use = np.where((self.d.flatten() >= d - h) & (self.d.flatten() < d))[0]
                    X = np.concatenate([self.d[use, :], np.ones((use.shape[0], 1))], axis = 1)
                    W = np.diag(self.weights[use]/np.mean(self.weights[use]))
                    Y = self.y[use, :]
                    est = np.array([[d, 1]]) @ np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y 
                    left_est[i, 0] = est[0, 0]
                
                for i in range(right_grid.shape[0]):
                    d = right_grid[i, 0]
                    use = np.where((self.d.flatten() <= d + h) & (self.d.flatten() > d))[0]
                    X = np.concatenate([self.d[use, :], np.ones((use.shape[0], 1))], axis = 1)
                    W = np.diag(self.weights[use]/np.mean(self.weights[use]))
                    Y = self.y[use, :]
                    est = np.array([[d, 1]]) @ np.linalg.inv(X.T @  W @ X) @ X.T @ W @ Y 
                    right_est[i, 0] = est[0, 0]

                mse = left_weights.T @ (left_est - left_true)**2 + right_weights.T @ (right_est - right_true)**2
                return mse

        res = minimize(bandwidth_loss, h0, method = "Nelder-Mead")
        if not res.success:
            warnings.warn("Warning: the bandwidth optimizer did not converge")
        self.bandwidth = res.x[0]

class Results():
    def __init__(self):
        self.params = None
        self.bse = None
        self.resid = None
        self.tvalues = None
        self.pvalues = None
        self.bandwidth = None
        self.predict = None
        self.n = None
        self.d_name = None
        self.y_name = None
        self.model = None
    
    def summary(self):
        if not type(self.bse) == type(None):
            left_ci = (1/np.sqrt(self.n - 3)) * t.ppf(0.025, self.n - 3) * self.bse['Treatment'] + self.params['Treatment']
            right_ci = (1/np.sqrt(self.n - 3)) * t.ppf(0.975, self.n - 3) * self.bse['Treatment'] + self.params['Treatment']
        
        length = 100
        print(self.model.center(length))
        print(''.center(length, '='))
        print(f'Dep. Variable:'.ljust(20) + str(self.y_name).rjust(28) + ''.center(4) + 
              f'Run Variable:'.ljust(20) + str(self.d_name).rjust(28))
        print(f'Model:'.ljust(20) + 'Local Linear'.rjust(28) + ''.center(4) + 
              f'No. Observations:'.ljust(20) + str(self.n).rjust(28))
        print(f'Covariance Type:'.ljust(20) + 'Heteroskedasticity-robust'.rjust(28) + ''.center(4) + 
              f'Bandwidth'.ljust(20) + f'{self.bandwidth:.3f}'.rjust(28))
        print(''.center(length, '='))
        print(''.center(40) + 'coef'.rjust(10) + 'std err'.rjust(10) + 't'.rjust(10) + 
              'P>|t|'.rjust(10) + '[0.025'.rjust(10) + '0.975]'.rjust(10))
        print(''.center(length, '-'))
        print('Treatment'.ljust(40) + f'{self.params['Treatment']:.3f}'.rjust(10) + f'{self.bse['Treatment']:.3f}'.rjust(10) +
              f'{self.tvalues['Treatment']:.3f}'.rjust(10) + f'{self.pvalues['Treatment']:.3f}'.rjust(10) +
              f'{left_ci:.3f}'.rjust(10) + f'{right_ci:.3f}'.rjust(10))
        print(''.center(length, '='))
        
class PDD():
    
    def __init__(self, outcome, runv, exog, ptreat, poutcome, cutoff, **kwargs):
        self.y_name = None
        self.d_name = None
        self.x_names = None
        self.z_names = None
        self.w_names = None
