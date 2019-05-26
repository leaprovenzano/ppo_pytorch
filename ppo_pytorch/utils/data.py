import keyword
from functools import singledispatch, update_wrapper
from collections import Mapping

import torch


class Attydict(dict):

    """It's a dictionary with attribute access.


    Examples:

        >>> from attydict import Attydict
        >>> data = Attydict(crab=2, stick=10)
        >>> data
        Attydict:
        {'crab': 2, 'stick': 10}
        >>> data.crab
        2
        >>> data.stick
        10
        >>> data.crab_stick = data.crab + data.stick
        >>> data
        Attydict:
        {'crab': 2, 'stick': 10, 'crab_stick': 12}

    """

    def __init__(self, **kwargs):
        for k in kwargs:
            self._assert_valid_key(k)
        super().__init__(**kwargs)

    def _raise_missing_attr(self, key):
        raise AttributeError(f'{self.__class__.__name__} object has no attribute {key}')

    def _assert_valid_key(self, key):
        if key in self.keys():
            return True
        if not isinstance(key, str):
            raise TypeError(f'AttrDict keys must be a string but got {key} of type {type(key).__name__}')

        elif not key.isidentifier():
            raise ValueError(f'Key {key} is not a valid python identifier')

        elif keyword.iskeyword(key):
            raise ValueError(f'Key {key} is a reserved keyword and may not be set.')

        elif key.startswith('_'):
            raise ValueError(f'Keys must not use private variable names, tried to set {key}')

        elif key in dir(self):
            raise ValueError(f'Key {key} conflicts with method nameand may not be set.')
        return True

    def __getattr__(self, key: str):
        if key in self:
            return self[key]
        raise self._raise_missing_attr(key)

    def __setitem__(self, key: str, value):
        self._assert_valid_key(key)
        super().__setitem__(key, value)

    def __setattr__(self, key: str, value):
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            self[key] = value

    def __repr__(self):
        return f'{self.__class__.__name__}:\n{super().__repr__()}'

    def __delattr__(self, key: str):
        if key in self:
            del self[key]
        else:
            self._raise_missing_attr(key)

    def _check_keys(self, other):
        for k, _ in other.items() if isinstance(other, Mapping) else other:
            self._assert_valid_key(k)
        return True

    def update(self, other=None, **kwargs):
        if other is not None:
            self._check_keys(other)
            super().update(other)
        if kwargs:
            self._check_keys(kwargs)
            super().update(**kwargs)




class TensorList(object):

    def __init__(self, *tensors):
        self.data = list(*tensors)

    def append(self, v):
        self.data.append(v)

    def stack(self):
        return torch.stack(self.data)

    def cat(self):
        return torch.cat(self.data)

    def __get__(self, instance, owner):
        print(instance, owner)
        return self.data

    def __set_name__(self, owner, name):
        self.name = name

    def __set__(self, instance, v):
        print(instance, v)
        self.data = v
