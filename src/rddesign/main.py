import numpy as np, pandas as pd


class RDD():
    
    def __init__(self, outcome, runv, exog, **kwargs):
        self.y_name = None
        self.d_name = None
        self.x_names = None
        
        if type(outcome) == pd.DataFrame:
            self.y = outcome.to_numpy()
            self.y_name = outcome.columns[0]
        elif type(outcome) == np.ndarray:
            self.y = outcome
            
        if type(runv) == pd.DataFrame:
            self.d = runv.to_numpy()
            self.d_name = runv.columns[0]
        elif type(runv) == np.ndarray:
            self.d = runv
            
        if type(exog) == pd.DataFrame:
            self.x = exog.to_numpy()
            self.x_names = exog.columns
        elif type(runv) == np.ndarray:
            self.x = exog
        

class PDD():
    
    def __init__(self, outcome, runv, exog, ptreat, poutcome):
        self.y_name = None
        self.d_name = None
        self.x_names = None
        self.z_names = None
        self.w_names = None
        
        if type(outcome) == pd.DataFrame:
            self.y = outcome.to_numpy()
            self.y_name = outcome.columns[0]
        elif type(outcome) == np.ndarray:
            self.y = outcome
            
        if type(runv) == pd.DataFrame:
            self.d = runv.to_numpy()
            self.d_name = runv.columns[0]
        elif type(runv) == np.ndarray:
            self.d = runv
            
        if type(exog) == pd.DataFrame:
            self.x = exog.to_numpy()
            self.x_names = exog.columns
        elif type(runv) == np.ndarray:
            self.x = exog
            
        if type(ptreat) == pd.DataFrame:
            self.z = ptreat.to_numpy()
            self.z_names = ptreat.columns
        elif type(ptreat) == np.ndarray:
            self.z = ptreat
            
        if type(poutcome) == pd.DataFrame:
            self.w = poutcome.to_numpy()
            self.w_names = poutcome.columns
        elif type(poutcome) == np.ndarray:
            self.w = poutcome
