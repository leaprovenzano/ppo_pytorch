import math 

import torch
from torch.distributions import constraints
from torch.distributions import Normal, Categorical


class TruncatedNormal(Normal):

    
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive,
                       'low': constraints.dependent, 'high': constraints.dependent}

    def __init__(self, loc, scale,  lower=-float('inf'), upper=float('inf')):
        super(TruncatedNormal, self).__init__(loc, scale)
        self.low, self.high = torch.ones_like(loc) * lower, torch.ones_like(loc) * upper

        self._delta = (self.cdf(self.high) - self.cdf(self.cdf(self.low)))
        self._delta[self.low>0] = -((self.cdf(-self.high) - self.cdf(self.cdf(-self.low))))[self.low>0]


    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            u = torch.rand_like(self.loc)
        return  self.icdf(self.cdf(self.low) +  u * self._delta)

    def log_prob_trunc(self, value, eps=1e-5):
        log_prob = self.log_prob(value) -  torch.log(self._delta)
        log_prob[(value < self.low) | (value > self.high)] = math.log(0 + eps)
        return log_prob
