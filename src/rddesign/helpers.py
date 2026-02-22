import torch
from scipy.stats import norm

def rectangle_kernel(u: torch.Tensor) -> torch.Tensor:
    u = torch.as_tensor(u, dtype=torch.float32)
    a = torch.abs(u)
    return torch.where(a <= 1.0, 1.0, torch.zeros_like(a)).to(torch.float32)

def triangular_kernel(u: torch.Tensor) -> torch.Tensor:
    u = torch.as_tensor(u, dtype=torch.float32)
    a = torch.abs(u)
    return torch.where(a <= 1.0, 1.0 - a, torch.zeros_like(a)).to(torch.float32)

def epanechnikov_kernel(u: torch.Tensor) -> torch.Tensor:
    u = torch.as_tensor(u, dtype=torch.float32)
    a = torch.abs(u)
    return torch.where(a <= 1.0, (3/4) * (1 - a**2), torch.zeros_like(a)).to(torch.float32)

class Results():
    def __init__(self, model, est, se, resid, bandwidth, n, predict, status):
        self.model = model
        self.est = est
        self.se = se
        self.resid = resid
        self.bandwidth = bandwidth
        self.n = n
        self.predict = predict
        self.status = status
        self.left_ci = norm.ppf(0.025) * self.se + self.est
        self.right_ci = norm.ppf(0.975) * self.se + self.est
        if est > 0:
            self.pvalue = 1 - norm.cdf(est/se)
        if est <= 0:
            self.pvalue = norm.cdf(est/se)
    
    def summary(self):
        output = self.__str__()
        print(output)
        
    def __str__(self):
        vartype = 'Robust Bias Correct'
        
        length = 100
        output = '\n' + (self.model).center(length) + '\n'
        output += ''.center(length, '=') + '\n'
        output += f'Var. type:'.ljust(20) + vartype.rjust(28) + ''.center(4) +\
                    f'No. Observations:'.ljust(20) + str(self.n).rjust(28) + '\n'
        output += f'Pos. Bandwidth:'.ljust(20) + f'{self.bandwidth['+']:.3f}'.rjust(28) + ''.center(4) +\
                    f'Neg. Bandwidth:'.ljust(20) + f'{self.bandwidth['-']:.3f}'.rjust(28) + '\n'
        output += ''.center(length, '=') + '\n'
        output += ''.center(40) + 'coef'.rjust(10) + 'std err'.rjust(10) + 't'.rjust(10) +\
              'p_1s'.rjust(10) + '[0.025'.rjust(10) + '0.975]'.rjust(10) + '\n'
        output += ''.center(length, '-') + '\n'
        output += 'Treatment'.ljust(40) + f'{self.est:.3f}'.rjust(10) + f'{self.se:.3f}'.rjust(10) +\
              f'{self.est/self.se:.3f}'.rjust(10) + f'{self.pvalue:.3f}'.rjust(10) +\
              f'{self.left_ci:.3f}'.rjust(10) + f'{self.right_ci:.3f}'.rjust(10) + '\n'
        output += ''.center(length, '=') + '\n'
        return output
