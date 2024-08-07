import numpy as np, pandas as pd, warnings
from scipy.optimize import minimize

class RDD():
    
    def __init__(self, outcome, runv, cutoff, exog = None, **kwargs):
        self.y_name = None
        self.d_name = None
        self.x_names = None
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
        
        self._find_bandwidth()
        
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
        pass

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
                left_true, right_true = self.y[use_left, :], self.y[use_right, :]
                
                for i in range(left_grid.shape[0]):
                    d = left_grid[i, 0]
                    use = np.where((self.d.flatten() >= d - h) & (self.d.flatten() < d))[0]
                    X = np.concatenate([self.d[use, :], np.ones((use.shape[0], 1))], axis = 1)
                    Y = self.y[use, :]
                    est = np.array([[d, 1]]) @ np.linalg.inv(X.T @ X) @ X.T @ Y 
                    left_est[i, 0] = est[0, 0]
                
                for i in range(right_grid.shape[0]):
                    d = right_grid[i, 0]
                    use = np.where((self.d.flatten() <= d + h) & (self.d.flatten() > d))[0]
                    X = np.concatenate([self.d[use, :], np.ones((use.shape[0], 1))], axis = 1)
                    Y = self.y[use, :]
                    est = np.array([[d, 1]]) @ np.linalg.inv(X.T @ X) @ X.T @ Y 
                    right_est[i, 0] = est[0, 0]

                obs_used = use_left.shape[0] + use_right.shape[0]
                mse = (1/obs_used) * (np.sum((left_est - left_true)**2) + np.sum((right_est - right_true)**2))
                return mse

        res = minimize(bandwidth_loss, h0, method = "Nelder-Mead")
        if not res.success:
            warnings.warn("Warning: the bandwidth optimizer did not converge")
        self.bandwidth = res.x

class PDD():
    
    def __init__(self, outcome, runv, exog, ptreat, poutcome, cutoff, **kwargs):
        self.y_name = None
        self.d_name = None
        self.x_names = None
        self.z_names = None
        self.w_names = None
