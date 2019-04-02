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
        self._last_n = n
        self._last_loc = loc
        self.loc = loc
        self._last_var = var
        self.var = var

    @property
    def mean(self):
        return self.loc

    @property
    def std(self):
        return np.sqrt(self.var)

    def update(self, x):
        self._last_loc = self.loc
        self._last_var = self.var
        self._last_n = self.n

        self.n += len(x)

        self.loc = ((self._last_loc * self._last_n) + x.sum()) / self.n

        v1 = ((x - self.loc)**2).sum() / self.n
        v2 = (((self._last_loc - self.loc)**2) + self._last_var) * (self._last_n / self.n)

        if v1 + v2 > 0:
            self.var = v1 + v2

