import torch
import numpy as np
import functools


from typing import Dict, Tuple, Sequence, Union


class MnMxRange(object):

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]

    def validate(self, vals):
        try:
            assert len(vals) == 2
            assert all(map(np.isfinite, vals))
            assert vals[0] < vals[1]
        except AssertionError:
            raise ValueError(f'Values are expected to be finite min value must be larger than max! recieved: {vals[0]}, {vals[1]}')

    def __set__(self, instance, value):
        self.validate(value)
        instance.__dict__[self.name] = value

    def __set_name__(self, owner, name):
        self.name = name


class MnMxScaler(object):

    scale_range = MnMxRange()
    input_range = MnMxRange()

    def __init__(self, scale_range: Tuple[float, float], input_range=(0, 1)):
        self.scale_range = scale_range
        self.input_range = input_range

    @staticmethod
    def standardize(x: Union[np.array, torch.tensor], mn: float, mx: float) -> Union[np.array, torch.tensor]:
        return (x - mn) / (mx - mn)

    def _scale(self, x: Union[np.array, torch.tensor], a: float, b: float)-> Union[np.array, torch.tensor]:
        (a_min, a_max), (b_min, b_max) = a, b
        return self.standardize(x, a_min, a_max) * (b_max - b_min) + b_min

    @torch.no_grad()
    def scale(self, x: Union[np.array, torch.tensor]):
        return self._scale(x, self.input_range, self.scale_range)

    @torch.no_grad()
    def inverse_scale(self, x: Union[np.array, torch.tensor]):
        return self._scale(x, self.scale_range, self.input_range)


def expand_dims(*shape):
    def expand_inner(f):
        @functools.wraps(f)
        def inner(*args, **kwargs):
            res = f(*args, **kwargs)
            if type(res) is tuple:
                return tuple(t.view(*shape) for t in res)
            return res.view(*shape)
        return inner
    return expand_inner


class classproperty(property):

    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)
