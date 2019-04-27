import numpy as np
import torch


class ValidationDescriptor(object):

    """Generic Basclass for descriptors that preform validation
    """

    def __init__(self, value=None):
        self.value = value

    def __get__(self, instance, owner):
        return self.value

    def __set__(self, instance, value):
        self._validate(value)
        self.value = value

    def __set_name__(self, owner, name):
        self.name = name

    def _validate(self, *values):
        raise NotImplementedError(f'the _validate method for {self.__class__.__name__} has not been implemented')

    def _raise_on_invalidation(self, msg, value=None):
        meta = f'-- recieved : {value}' if value is not None else ''
        raise ValueError(f'{self.name} : {msg}{meta}')
