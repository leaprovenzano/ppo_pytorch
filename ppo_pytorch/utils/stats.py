import numpy as np

class RunningStat(object):
    """utility class for track running mean and varience.

    Attributes:
        loc (float): on init represents a starting center otherwise it is the 
            this attr stores current running mean.
        n (int): number of values seen. 
        var (TYPE): on init represents a starting varience otherwise it is the
            running varience
    """

    def __init__(self, loc=0., var=1, n=0):
        self.n = n
        self.loc = loc
        self.var = var

    @property
    def mean(self):
        return self.loc

    @property
    def std(self):
        return np.sqrt(self.var)

    def update(self, x):
        last_loc = self.loc
        last_var = self.var
        last_n = self.n

        self.n += len(x)
        self.loc = ((self.loc * last_n) + x.sum()) / self.n

        v1 = ((x - self.loc) ** 2).sum() / self.n
        v2 = (((last_loc - self.loc) ** 2) + last_var) * (last_n / self.n)

        if v1 + v2 > 0:
            self.var = v1 + v2