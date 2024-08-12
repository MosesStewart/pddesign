import pandas as pd, numpy as np
from scipy.stats import t, chi2

class FuzzyResults():
    def __init__(self):
        self.params = None
        self.bse = None
        self.pvalues = None
        self.bandwidth = None
        self.n = None
        self.order = None
        self.d_name = None
        self.y_name = None
        self.model = None
        self.replicates = None
        self.nreps = None
        
    def summary(self):
        output = self.__str__()
        print(output)
    
    def __str__(self):
        length = 100
        output = '\n' + ('Fuzzy ' + self.model.capitalize()).center(length) + '\n'
        output += ''.center(length, '=') + '\n'
        output += f'Dep. Variable:'.ljust(20) + str(self.y_name).rjust(28) + ''.center(4) + \
                  f'Run. Variable:'.ljust(20) + str(self.d_name).rjust(28) + '\n'
        output += f'Model:'.ljust(20) + ('Fuzzy ' + self.model.capitalize()).rjust(28) + ''.center(4) + \
                  f'No. Observations:'.ljust(20) + str(self.n).rjust(28)+ '\n'
        if self.model == 'local linear':
            output += f'Variance Type:'.ljust(20) + 'Bayes bootstrap'.rjust(28) + ''.center(4) + \
                      f'Bandwidth:'.ljust(20) + f'{self.bandwidth:.3f}'.rjust(28) + '\n'
        elif self.model == 'polynomial':
            output += f'Variance Type:'.ljust(20) + 'Bayes bootstrap'.rjust(28) + ''.center(4) +\
                      f'Polynomial Order:'.ljust(20) + f'{self.order:d}'.rjust(28) + '\n'
        output += ''.center(length, '=') + '\n'
        output += ''.center(48) + 'coef'.rjust(13) + 'std err'.rjust(13) +\
                  '[0.025'.rjust(13) + '0.975]'.rjust(13) + '\n'
        output += ''.center(length, '-') + '\n'
        if type(self.replicates) == type(None):
            output += 'Treatment'.ljust(48) + f'{self.params['Treatment']:.3f}'.rjust(13) + str(None).rjust(13) +\
                  str(None).rjust(13) + str(None).rjust(13) + '\n'
        else:
            output += 'Treatment'.ljust(48) + f'{self.params['Treatment']:.3f}'.rjust(13) + \
                  f'{self.bse['Treatment']:.3f}'.rjust(13) + f'{np.quantile(self.replicates, 0.025):.3f}'.rjust(13) +\
                  f'{np.quantile(self.replicates, 0.975):.3f}'.rjust(13) + '\n'
        output += ''.center(length, '=') + '\n'
        return output
        
class SharpResults():
    def __init__(self):
        self.params = None
        self.bse = None
        self.resid = None
        self.tvalues = None
        self.pvalues = None
        self.bandwidth = None
        self.replicates = None
        self.n = None
        self.order = None
        self.d_name = None
        self.y_name = None
        self.model = None
    
    def summary(self):
        output = self.__str__()
        print(output)
        
    def __str__(self):
        if not type(self.bse) == type(None) and not type(self.replicates) == type(None):
            left_ci = np.quantile(self.replicates, 0.025)
            right_ci = np.quantile(self.replicates, 0.975)
            vartype = 'Bayes bootstrap'
        elif not type(self.bse) == type(None):
            left_ci = t.ppf(0.025, self.n - 4) * self.bse['Treatment'] + self.params['Treatment']
            right_ci = t.ppf(0.975, self.n - 4) * self.bse['Treatment'] + self.params['Treatment']
            vartype = 'Heteroskedasticity robust'
        
        length = 100
        output = '\n' + ('Sharp ' + self.model.capitalize()).center(length) + '\n'
        output += ''.center(length, '=') + '\n'
        output += f'Dep. Variable:'.ljust(20) + str(self.y_name).rjust(28) + ''.center(4) +\
                  f'Run. Variable:'.ljust(20) + str(self.d_name).rjust(28) + '\n'
        output += f'Model:'.ljust(20) + ('Sharp ' + self.model.capitalize()).rjust(28) + ''.center(4) +\
                  f'No. Observations:'.ljust(20) + str(self.n).rjust(28) + '\n'
        if self.model == 'local linear':
            output += f'Covariance Type:'.ljust(20) + vartype.rjust(28) + ''.center(4) +\
                      f'Bandwidth:'.ljust(20) + f'{self.bandwidth:.3f}'.rjust(28) + '\n'
        elif self.model == 'polynomial':
            output += f'Covariance Type:'.ljust(20) + vartype.rjust(28) + ''.center(4) +\
                      f'Polynomial Order:'.ljust(20) + f'{self.order:d}'.rjust(28) + '\n'
        output += ''.center(length, '=') + '\n'
        output += ''.center(40) + 'coef'.rjust(10) + 'std err'.rjust(10) + 't'.rjust(10) +\
              'P>|t|'.rjust(10) + '[0.025'.rjust(10) + '0.975]'.rjust(10) + '\n'
        output += ''.center(length, '-') + '\n'
        output += 'Treatment'.ljust(40) + f'{self.params['Treatment']:.3f}'.rjust(10) + f'{self.bse['Treatment']:.3f}'.rjust(10) +\
              f'{self.tvalues['Treatment']:.3f}'.rjust(10) + f'{self.pvalues['Treatment']:.3f}'.rjust(10) +\
              f'{left_ci:.3f}'.rjust(10) + f'{right_ci:.3f}'.rjust(10) + '\n'
        output += ''.center(length, '=') + '\n'
        return output

class HTestResults():
    def __init__(self):
        self.params = None
        self.bse = None
        self.n = None
        self.order = None
        self.d_name = None
        self.model = None
        self.waldstat = None
        self.bandwidth = None
        self.pvalue = None
    
    def summary(self):
        output = self.__str__()
        print(output)
        
    def __str__(self):
        length = 100
        output = '\n' + (self.model.capitalize() + " Continuity Wald Test").center(length) + '\n'
        output += ''.center(length, '=') + '\n'
        output += f'Model:'.ljust(20) + self.model.rjust(28) + ''.center(4) +\
                  f'Run. Variable:'.ljust(20) + str(self.d_name).rjust(28) + '\n'
        output += f'Covariance Type:'.ljust(20) + 'Heteroskedasticity-robust'.rjust(28) + ''.center(4) +\
                  f'No. Observations:'.ljust(20) + str(self.n).rjust(28) + '\n'
        output += f'Joint pvalue:'.ljust(20) + f'{self.pvalue:.3f}'.rjust(28) + ''.center(4) +\
                  f''.ljust(20) + f''.rjust(28) + '\n'
        output += ''.center(length, '=')
        output += 'Dep. Variable'.ljust(20) + 'Treatment ...'.rjust(20) + 'coef'.rjust(12) + 'std err'.rjust(12) + 'waldstat'.rjust(12) +\
                  'order'.rjust(12) + 'bandwidth'.rjust(12) + '\n'
        output += ''.center(length, '-') + '\n'
        for var in self.params.index:
            if self.model == 'local linear':
                output += str(var).ljust(40) + f'{self.params[var]:.3f}'.rjust(12) + f'{self.bse[var]:.3f}'.rjust(12) +\
                    f'{self.waldstat[var]:.3f}'.rjust(12) + str(None).rjust(12) +\
                    f'{self.bandwidth[var]:.3f}'.rjust(12) + '\n'
            elif self.model == 'polynomial':
                output += str(var).ljust(40) + f'{self.params[var]:.3f}'.rjust(12) + f'{self.bse[var]:.3f}'.rjust(12) +\
                    f'{self.waldstat[var]:.3f}'.rjust(12) + f'{self.order[var]:d}'.rjust(12) +\
                    str(None).rjust(12) + '\n'
        output += ''.center(length, '=') + '\n'
