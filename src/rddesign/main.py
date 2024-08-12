import numpy as np, pandas as pd, warnings
from scipy.optimize import minimize, Bounds
from scipy.stats import t, chi2
from helpers import *

class RDD():
    def __init__(self, outcome, runv, cutoff, treatment = None, exog = None, weights = None):
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
            if exog.ndim == 1:
                self.x = exog.reshape((self.n, 1))
            else:
                self.x = exog
        else:
            self.x = None
        
        if type(treatment) == pd.DataFrame:
            self.a = treatment.to_numpy()
            self.a_names = treatment.columns
        elif type(treatment) == np.ndarray:
            self.a = np.reshape(treatment, (self.n, 1))
        else:
            self.a = np.where(self.d.flatten() > cutoff, 1, 0)
        
        if type(weights) == pd.DataFrame:
            wgts = weights.to_numpy()
            self.weights = wgts/np.mean(wgts)
        elif type(weights) == np.ndarray:
            self.weights = weights.flatten()/np.mean(weights.flatten())
        else:
            self.weights = np.ones(self.n)
        
    def fit(self, model = "local linear", design = 'sharp', bootstrap = False, **kwargs):
        if model == "local linear":
            res = self.fit_local_lin(design = design, **kwargs)
            if bootstrap:
                res.replicates = self.bootstrap(model = "local linear", design = design,
                                                bandwidth = res.bandwidth, **kwargs)
                res.bse = pd.Series({'Treatment': np.std(res.replicates)}, name = 'Standard error')
        elif model == "polynomial":
            res = self.fit_polynomial(design = design, **kwargs)
            if bootstrap:
                res.replicates = self.bootstrap(model = "polynomial", design = design,
                                                order = res.order,  **kwargs)
                res.bse = pd.Series({'Treatment': np.std(res.replicates)}, name = 'Standard error')
        return res
        
    def fit_polynomial(self, order = None, design = 'sharp', **kwargs):
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
        if not type(self.x) == type(None):
            Y = self.y
            x = self.x
            X = np.concatenate([x, np.ones((self.n, 1))], axis = 1)
            Y = Y - X @ np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y
        else:
            Y = self.y
        X = np.concatenate([isright, np.ones((self.n, 1))], axis = 1)
        for pow in range(1, self.order + 1):
            X = np.concatenate([X, (self.d - self.cutoff)**pow, isright * (self.d - self.cutoff)**pow], axis = 1)
        coefs = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y
        def predict(runv):
            runv = runv.reshape((runv.shape[0], 1))
            isright = np.where(runv >= self.cutoff, 1, 0)
            Xnew = np.concatenate([isright, np.ones((runv.shape[0], 1))], axis = 1)
            for pow in range(1, self.order + 1):
                Xnew = np.concatenate([Xnew, (runv - self.cutoff)**pow, isright * (runv - self.cutoff)**pow], axis = 1)
            Yhat = Xnew @ coefs
            return Yhat

        if design == 'sharp':
            r = Y - X @ coefs
            M = np.diag(r.flatten()**2)
            varcov = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ M @ W.T @ X @ np.linalg.inv(X.T @ W @ X)
            tstat = coefs[0, 0]/np.sqrt(varcov[0, 0])
            pval = min(t.cdf(tstat, df = self.n - 4) + (1 - t.cdf(-tstat, df = self.n - 4)), 
                    t.cdf(-tstat, df = self.n - 4) + (1 - t.cdf(tstat, df = self.n - 4)))
            results = SharpResults()
            results.params = pd.Series({'Treatment': coefs[0, 0]}, name = 'Estimated effect')
            results.bse = pd.Series({'Treatment': np.sqrt(varcov[0, 0])}, name = 'Standard error')
            results.resid = r.flatten()
            results.tvalues = pd.Series({'Treatment': tstat}, name = 't-statistic')
            results.pvalues = pd.Series({'Treatment': pval}, name = 'p-value > |t|')
        elif design == 'fuzzy':
            A = self.a
            Ahat = X @ np.linalg.inv(X.T @ W @ X) @ X.T @ W @ A
            X[:, 0] = Ahat.flatten()
            fuz_coefs = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y
            results = FuzzyResults()
            results.params = pd.Series({'Treatment': fuz_coefs[0, 0]}, name = 'Estimated effect')
        results.predict = predict
        results.order = self.order
        results.model = 'polynomial'
        results.d_name = self.d_name
        results.y_name = self.y_name
        results.n = self.n
        return results
        
    def fit_local_lin(self, bandwidth = None, design = 'sharp', **kwargs):
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
        if not type(self.x) == type(None):
            Y = self.y[use, :]
            x = self.x[use, :]
            X = np.concatenate([x, np.ones((n, 1))], axis = 1)
            Y = Y - X @ np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y
        else:
            Y = self.y[use,:]
        X = np.concatenate([isright, np.ones((n, 1)), d, isright * d], axis = 1)
        coefs = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y
        def predict(runv):
            runv = runv.reshape((runv.shape[0], 1))
            isright = np.where(runv >= self.cutoff, 1, 0)
            Xnew = np.concatenate([isright, np.ones((runv.shape[0], 1)), runv, isright * runv], axis = 1)
            Yhat = Xnew @ coefs
            return Yhat
        
        if design == 'sharp':
            r = Y - X @ coefs
            M = np.diag(r.flatten()**2)
            varcov = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ M @ W.T @ X @ np.linalg.inv(X.T @ W @ X)
            tstat = coefs[0, 0]/np.sqrt(varcov[0, 0])
            pval = min(t.cdf(tstat, n - 4) + (1 - t.cdf(-tstat, n - 4)), t.cdf(-tstat, n - 4) + (1 - t.cdf(tstat, n - 4)))
            results = SharpResults()
            results.params = pd.Series({'Treatment': coefs[0, 0]}, name = 'Estimated effect')
            results.bse = pd.Series({'Treatment': np.sqrt(varcov[0, 0])}, name = 'Standard error')
            results.resid = r.flatten()
            results.tvalues = pd.Series({'Treatment': tstat}, name = 't-statistic')
            results.pvalues = pd.Series({'Treatment': pval}, name = 'p-value > |t|')
        elif design == 'fuzzy':
            A = self.a[use, :]
            Ahat = X @ np.linalg.inv(X.T @ W @ X) @ X.T @ W @ A
            X[:, 0] = Ahat.flatten()
            fuz_coefs = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y
            results = FuzzyResults()
            results.params = pd.Series({'Treatment': fuz_coefs[0, 0]}, name = 'Estimated effect')
        results.predict = predict
        results.bandwidth = self.bandwidth
        results.model = 'local linear'
        results.d_name = self.d_name
        results.y_name = self.y_name
        results.n = n
        return results

    def continuity_test(self, model = "local linear", residualize = False, **kwargs):
        '''
        Following Lee and Lemieux (2010), estimates the treatment effect using
            exogenous variables as placebo outcomes, and performs a joint Wald
            hypothesis test that the effects are not equal to zero. A higher
            pvalue is stronger evidence of no discontinuity
        :param model: One of either "local linear" or "polynomial" to determine
            which model to use in estimation
        :param residualize: Before estimation, whether to residualize the placebo
            outcomes using the other exogenous variables
        '''
        if type(self.x) == type(None):
            raise ValueError("You must include exogenous variables in the model to perform a continuity test. ")
        # Save original specification
        y = self.y
        x = self.x
        
        res = HTestResults()
        res.waldstat, res.params, res.bse = pd.Series(), pd.Series(), pd.Series()
        res.order, res.bandwidth = pd.Series(), pd.Series()
        test_stats = np.empty(x.shape[1])
        for var in range(self.x.shape[1]):
            placebo = x[:, var]
            self.y = placebo[:, None]
            if residualize == True:
                self.x = np.concatenate([x[:, :var], x[:, (var + 1):]], axis = 1)
            else:
                self.x = None
            if model == "local linear":
                xresult = self.fit_local_lin(**kwargs)
            elif model == "polynomial":
                xresult = self.fit_polynomial(**kwargs)
            test_stats[var] = (xresult.params["Treatment"]/xresult.bse["Treatment"])**2
            if type(self.x_names) == type(None):
                res.waldstat[var] = test_stats[var]
                res.params[var], res.bse[var ]= xresult.params["Treatment"], xresult.bse["Treatment"]
                res.bandwidth[var], res.order[var] = xresult.bandwidth, xresult.order
            else:
                res.waldstat[self.x_names[var]] = test_stats[var]
                res.params[self.x_names[var]], res.bse[self.x_names[var]] = xresult.params["Treatment"], xresult.bse["Treatment"]
                res.bandwidth[self.x_names[var]], res.order[self.x_names[var]] = xresult.bandwidth, xresult.order

        self.y = y
        self.x = x
        pval = 1 - chi2.cdf(np.sum(test_stats), df = test_stats.shape[0])
        res.pvalue, res.model, res.n = pval, model, self.n
        return res
    
    def bootstrap(self, nreps = 200, seed = 10042002, model = 'local linear', design = 'fuzzy', **kwargs):
        rng = np.random.default_rng(seed = seed)
        w0 = self.weights
        weights = rng.exponential(1, size = (self.n, nreps))
        weights = weights/np.sum(weights, axis = 0) * w0[:, None]
        reps = np.empty(nreps)
        for i in range(nreps):
            self.weights = weights[:, i]
            if model == 'local linear':
                res = self.fit_local_lin(design = design, **kwargs)
            elif model == 'polynomial':
                res = self.fit_polynomial(design = design, **kwargs)
            reps[i] = res.params['Treatment']
        self.weights = w0
        return reps
    
    def _find_order(self, optimization = 'aic', **kwargs):
        if optimization.lower() == 'significance bins':
            self._find_order_sb(**kwargs)
        elif optimization.lower() == 'aic':
            self._find_order_aic()
    
    def _find_order_sb(self, nbins = None, pval = 0.05):
        '''
        Following Lee and Lemieux (2010), finds the lowest polynomial order
            such that the null hypothesis that the coefficients for the
            indicator of being in a bin is zero
        
        :param nbins: Number of bins to use in wald test
        :param pval: Probability tolerance for wald test
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
        if not type(self.x) == type(None):
            Y = self.y
            x = self.x
            X = np.concatenate([x, np.ones((self.n, 1))], axis = 1)
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
        if not type(self.x) == type(None):
            Y = self.y
            x = self.x
            X = np.concatenate([x, np.ones((self.n, 1))], axis = 1)
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
            prev_loss[1] = prev_loss[0]
            prev_loss[0] = cur_loss
            cur_loss = aic_loss(order)
        self.order = order - 2
        
    def _find_bandwidth(self, h0 = None):
        '''
        Uses the leave-one-out procedure from Jens Ludwig and
            Douglas Miller (2007) and Imbens and Lemieux (2008)
            to compute optimal bandwidth for local linear regression
        
        :param h0: The initial bandiwdth used in the optimization
        '''
        d_sorted = np.sort(self.d)
        hmin = 2 * np.mean(d_sorted[1:] - d_sorted[:-1])
        hmax = max(np.median(self.d) - np.min(self.d), np.max(self.d) - np.median(self.d))
        bounds = Bounds(lb = hmin, ub = hmax)
        if h0 == None:
            h0 = 1/2 * (np.quantile(self.d, 0.75) - np.quantile(self.d, 0.25))
        
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
                    if not type(self.x) == type(None):
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
                    if not type(self.x) == type(None):
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

        res = minimize(bandwidth_loss, h0, method = "Nelder-Mead", bounds = bounds)
        if not res.success:
            warnings.warn("Warning: the bandwidth optimizer did not converge")
        self.bandwidth = res.x[0]

class PDD():
    def __init__(self, outcome, runv, exog, ptreat, poutcome, cutoff, **kwargs):
        self.y_name = None
        self.d_name = None
        self.x_names = None
        self.z_names = None
        self.w_names = None
