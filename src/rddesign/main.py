import numpy as np, pandas as pd, warnings
from scipy.optimize import minimize
from scipy.stats import t, chi2

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
        
    def fit_polynomial(self, order = None, **kwargs):
        '''
        Fits the model with a polynomial
        
        :param order: the order of the polynomial function
        '''
        if order == None:
            self._find_order(**kwargs)
        else:
            self.order = order
        
        isright = np.where(self.d >= self.cutoff, 1, 0)
        W = np.diag(self.weights)
        if not self.x == None:
            Y = self.y
            x = self.x
            X = np.concatenate([isright, x, np.ones((self.n, 1)), isright * x], axis = 1)
            Y = Y - X @ np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y
        else:
            Y = self.y
        X = np.concatenate([isright, np.ones((self.n, 1))], axis = 1)
        for pow in range(1, self.order + 1):
            X = np.concatenate([X, (self.d - self.cutoff)**pow, isright * (self.d - self.cutoff)**pow], axis = 1)
        coefs = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y
        r = Y - X @ coefs
        M = np.diag(r.flatten()**2)
        varcov = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ M @ W.T @ X @ np.linalg.inv(X.T @ W @ X)
        tstat = np.sqrt(self.n - 4) * coefs[0, 0]/np.sqrt(varcov[0, 0])
        pval = min(t.cdf(tstat, self.n - 4) + (1 - t.cdf(-tstat, self.n - 4)), t.cdf(-tstat, self.n - 4) + (1 - t.cdf(tstat, self.n - 4)))
        
        results = Results()
        results.params = pd.Series({'Treatment': coefs[0, 0]}, name = 'Estimated effect')
        results.bse = pd.Series({'Treatment': np.sqrt(varcov[0, 0])}, name = 'Standard error')
        results.resid = r.flatten()
        results.tvalues = pd.Series({'Treatment': tstat}, name = 't-statistic')
        results.pvalues = pd.Series({'Treatment': pval}, name = 'p-value != 0')
        results.order = self.order
        results.model = 'Sharp Polynomial Regression'
        results.d_name = self.d_name
        results.y_name = self.y_name
        results.n = self.n
        return results
        
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
        W = np.diag(self.weights[use])
        
        # Residualize Y with respect to exogenous covariates
        if not self.x == None:
            Y = self.y[use, :]
            x = self.x[use, :]
            X = np.concatenate([isright, x, np.ones((n, 1)), isright * x], axis = 1)
            Y = Y - X @ np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y
        else:
            Y = self.y[use,:]
        X = np.concatenate([isright, np.ones((n, 1)), d, isright * d], axis = 1)
        coefs = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y
        
        r = Y - X @ coefs
        M = np.diag(r.flatten()**2)
        varcov = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ M @ W.T @ X @ np.linalg.inv(X.T @ W @ X)
        tstat = np.sqrt(n - 4) * coefs[0, 0]/np.sqrt(varcov[0, 0])
        pval = min(t.cdf(tstat, n - 4) + (1 - t.cdf(-tstat, n - 4)), t.cdf(-tstat, n - 4) + (1 - t.cdf(tstat, n - 4)))
        
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

    def _find_order(self, method = 'significance bins', **kwargs):
        if method.lower() == 'significance bins':
            self._find_order_sb()
        elif method.lower() == 'aic':
            self._find_order_aic()
    
    def _find_order_sb(self, nbins = None, pval = 0.05):
        '''
        Following Lee and Lemieux (2010), finds the lowest polynomial order
            such that the null hypothesis that the coefficients for the
            indicator of being in a bin is zero
        
        :param nbins: Number of bins to use in chi-2 test
        :param pval: Probability tolerance for chi-2 test
        '''
        if nbins == None:
            nbins = int(self.n/20)
        sorted_d = np.sort(self.d.flatten())
        indicators = []
        for bin in range(1, nbins - 1):
            in_bin = np.where((self.d >= sorted_d[bin * nbins]) & (self.d < sorted_d[(bin + 1) * nbins]), 1, 0)
            indicators.append(in_bin)
        indicators = np.concatenate(indicators, axis = 1)
        isright = np.where(self.d >= self.cutoff, 1, 0)
        W = np.diag(self.weights)
        if not self.x == None:
            Y = self.y
            x = self.x
            X = np.concatenate([isright, x, np.ones((self.n, 1)), isright * x], axis = 1)
            Y = Y - X @ np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y
        else:
            Y = self.y
        
        order = 1
        prob = -np.inf
        
        while prob < pval:
            order += 1
            X = np.concatenate([isright, np.ones((self.n, 1)), indicators], axis = 1)
            for pow in range(1, order + 1):
                X = np.concatenate([(self.d - self.cutoff)**pow, isright * (self.d - self.cutoff)**pow, X], axis = 1)
            coefs = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y
            r = Y - X @ coefs
            M = np.diag(r.flatten()**2)
            varcov = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ M @ W.T @ X @ np.linalg.inv(X.T @ W @ X)
            bin_coefs = coefs[-indicators.shape[0]:,:].flatten()
            bin_ses = np.sqrt(np.diag(varcov)[-indicators.shape[0]:])
            prob = 1 - chi2.cdf(np.sum((bin_coefs/bin_ses)**2), df = bin_coefs.shape[0])
        order -= 1
        self.order = order
    
    def _find_order_aic(self):
        '''
        Uses cross-validation procedure following Dan A. Black, Jose 
            Galdo, and Smith (2007) to choose order of polynomial with
            Akaike information criterion (AIC) loss
        '''
        isright = np.where(self.d >= self.cutoff, 1, 0)
        W = np.diag(self.weights)
        if not self.x == None:
            Y = self.y
            x = self.x
            X = np.concatenate([isright, x, np.ones((self.n, 1)), isright * x], axis = 1)
            Y = Y - X @ np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y
        else:
            Y = self.y
        
        def aic_loss(order):
            X = np.concatenate([isright, np.ones((self.n, 1))], axis = 1)
            for pow in range(1, int(order + 1)):
                X = np.concatenate([X, (self.d - self.cutoff)**pow, isright * (self.d - self.cutoff)**pow], axis = 1)
            coefs = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y
            r = Y - X @ coefs
            loss = self.n * np.log(np.sum(r**2)) + 2 * order
            return loss
        
        order = 1
        prev_loss = np.array([np.inf, np.inf])
        cur_loss = aic_loss(order)
        while prev_loss[1] >= prev_loss[0] and prev_loss[1] >= cur_loss:
            order += 1
            prev_loss[0] = cur_loss
            prev_loss[1] = prev_loss[0]
            cur_loss = aic_loss(order)
        
        self.order = order - 2
        
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
                    W = np.diag(self.weights[use]/np.mean(self.weights[use]))
                    if not self.x == None:
                        Y = self.y[use, :]
                        x = self.x[use, :]
                        X = np.concatenate([x, np.ones((use.shape[0], 1)),], axis = 1)
                        Y = Y - X @ np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y
                    else:
                        Y = self.y[use,:]
                    X = np.concatenate([self.d[use, :], np.ones((use.shape[0], 1))], axis = 1)
                    est = np.array([[d, 1]]) @ np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y 
                    left_est[i, 0] = est[0, 0]
                
                for i in range(right_grid.shape[0]):
                    d = right_grid[i, 0]
                    use = np.where((self.d.flatten() <= d + h) & (self.d.flatten() > d))[0]
                    W = np.diag(self.weights[use]/np.mean(self.weights[use]))
                    if not self.x == None:
                        Y = self.y[use, :]
                        x = self.x[use, :]
                        X = np.concatenate([x, np.ones((use.shape[0], 1)),], axis = 1)
                        Y = Y - X @ np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y
                    else:
                        Y = self.y[use,:]
                    X = np.concatenate([self.d[use, :], np.ones((use.shape[0], 1))], axis = 1)
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
        self.order = None
        self.d_name = None
        self.y_name = None
        self.model = None
    
    def summary(self):
        if not type(self.bse) == type(None):
            left_ci = t.ppf(0.025, self.n - 4) * self.bse['Treatment'] + self.params['Treatment']
            right_ci = t.ppf(0.975, self.n - 4) * self.bse['Treatment'] + self.params['Treatment']
        
        length = 100
        print(self.model.center(length))
        print(''.center(length, '='))
        print(f'Dep. Variable:'.ljust(20) + str(self.y_name).rjust(28) + ''.center(4) + 
              f'Run. Variable:'.ljust(20) + str(self.d_name).rjust(28))
        print(f'Model:'.ljust(20) + 'Local Linear'.rjust(28) + ''.center(4) + 
              f'No. Observations:'.ljust(20) + str(self.n).rjust(28))
        if self.model == 'Sharp Local Linear Regression':
            print(f'Covariance Type:'.ljust(20) + 'Heteroskedasticity-robust'.rjust(28) + ''.center(4) + 
              f'Bandwidth:'.ljust(20) + f'{self.bandwidth:.3f}'.rjust(28))
        elif self.model == 'Sharp Polynomial Regression':
            print(f'Covariance Type:'.ljust(20) + 'Heteroskedasticity-robust'.rjust(28) + ''.center(4) + 
              f'Polynomial Order:'.ljust(20) + f'{self.order:d}'.rjust(28))
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
