import pandas as pd, numpy as np
from scipy.stats import t, chi2, norm
        
class PDDResults():
    def __init__(self, est, se, resid, bandwidth, n, predict, status):
        self.est = est
        self.se = se
        self.resid = resid
        self.bandwidth = bandwidth
        self.n = n
        self.predict = predict
        self.status = status
        if est > 0:
            self.pvalue = 1 - t.ppf(est/se)
        if est <= 0:
            self.pvalue = t.ppf(est/se)
    
    def summary(self):
        output = self.__str__()
        print(output)
        
    def __str__(self):
        left_ci = t.ppf(0.025, self.n - 4) * self.se + self.est
        right_ci = t.ppf(0.975, self.n - 4) * self.se + self.est
        vartype = 'Robust Bias Corr.'
        
        length = 100
        output = '\n' + ('Placebo Discontinuity Design').center(length) + '\n'
        output += ''.center(length, '=') + '\n'
        output += f'Var. type:'.ljust(20) + vartype.rjust(28) + ''.center(4) +\
                    f'No. Observations:'.ljust(20) + str(self.n).rjust(28) + '\n'
        output += ''.center(length, '=') + '\n'
        output += ''.center(40) + 'coef'.rjust(10) + 'std err'.rjust(10) + 't'.rjust(10) +\
              'P>|t|'.rjust(10) + '[0.025'.rjust(10) + '0.975]'.rjust(10) + '\n'
        output += ''.center(length, '-') + '\n'
        output += 'Treatment'.ljust(40) + f'{self.est:.3f}'.rjust(10) + f'{self.se:.3f}'.rjust(10) +\
              f'{self.est/self.se:.3f}'.rjust(10) + f'{self.pvalue:.3f}'.rjust(10) +\
              f'{left_ci:.3f}'.rjust(10) + f'{right_ci:.3f}'.rjust(10) + '\n'
        output += ''.center(length, '=') + '\n'
        return output
