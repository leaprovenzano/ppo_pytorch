import pytest
import numpy as np

from hypothesis import given
from hypothesis.strategies import (builds, integers, booleans, deferred, lists, tuples, floats, composite, one_of)
from hypothesis.extra.numpy import arrays, array_shapes
from .strategies.torchtensors import float_tensors

from ppo_pytorch.utils.scalers import MinMaxScaler


@composite
def valid_min_max_tensor_inp(draw):
    floatvals = floats(min_value=-100, max_value=100)
    a_min = draw(floatvals)
    a_max = draw(floats(min_value=a_min + 1, max_value=a_min + 100))
    b_min = draw(floatvals)
    b_max = draw(floats(min_value=b_min + 1, max_value=b_min + 100))

    scaleto = (a_min, a_max)
    scalefrom = (b_min, b_max)

    inp = draw(float_tensors(shape=array_shapes(min_dims=1, max_dims=3, min_side=2, max_side=10),
                             unique=True, elements=floats(min_value=b_min, max_value=b_max)))
    return scalefrom, scaleto, inp


@composite
def valid_min_max_numpy_inp(draw):
    floatvals = floats(min_value=-100, max_value=100)
    a_min = draw(floatvals)
    a_max = draw(floats(min_value=a_min + 1, max_value=a_min + 100))
    b_min = draw(floatvals)
    b_max = draw(floats(min_value=b_min + 1, max_value=b_min + 100))

    scaleto = (a_min, a_max)
    scalefrom = (b_min, b_max)

    inp = draw(arrays(dtype='float', shape=array_shapes(min_dims=1, max_dims=3, min_side=2, max_side=10),
                      unique=True, elements=floats(min_value=b_min, max_value=b_max)))
    return scalefrom, scaleto, inp


@composite
def params_as_numpy_arrays(draw):
    floatvals = floats(min_value=.001, max_value=100)
    a_min = draw(arrays(dtype='float32', shape=array_shapes(min_dims=1, max_dims=1, min_side=2, max_side=100), elements=floats(min_value=-100, max_value=100)))
    a_max = a_min + draw(floatvals)
    b_min = draw(arrays(dtype='float', shape=a_min.shape, elements=floats(min_value=-100, max_value=100)))
    b_max = b_min + draw(floatvals)

    scaleto = (a_min, a_max)
    scalefrom = (b_min, b_max)
    inp_shape = draw(one_of(tuples(), tuples(integers(1, 1000))).map(lambda x: x + a_min.shape))
    inp = np.random.uniform(*scalefrom, size=inp_shape)
    return scalefrom, scaleto, inp


class TestMinMaxScaler(object):

    def assert_reversable(self, scalefrom, scaleto, inp):
        amin, amax = scaleto
        bmin, bmax = scalefrom

        scaler = MinMaxScaler(scalefrom, scaleto)
        scaled = scaler.scale(inp)
        unscaled = scaler.inverse_scale(scaled)
        np.testing.assert_allclose(unscaled, inp, atol=1e-3)

    @given(params_as_numpy_arrays())
    def test_min_max_as_numpy_arrays(self, strat):
        self.assert_reversable(*strat)

    @given(valid_min_max_numpy_inp())
    def test_nparrays(self, strat):
        self.assert_reversable(*strat)

    @given(valid_min_max_tensor_inp())
    def test_tensors(self, strat):
        self.assert_reversable(*strat)
