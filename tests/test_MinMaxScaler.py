import pytest
import numpy as np
from ppo_pytorch.utils.general import MinMaxScaler

from hypothesis import given
from hypothesis.strategies import (builds, integers, booleans, deferred, lists, tuples, floats, composite, one_of)
from hypothesis.extra.numpy import arrays, array_shapes
from .strategies.torchtensors import float_tensors


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
    return scaleto, scalefrom, inp


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
    return scaleto, scalefrom, inp


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
    return scaleto, scalefrom, inp


def _test_MnMxScaler(strat):
    scaleto, scalefrom, inp = strat
    amin, amax = scaleto
    bmin, bmax = scalefrom

    scaler = MinMaxScaler(scaleto, scalefrom)
    scaled = scaler.scale(inp)
    unscaled = scaler.inverse_scale(scaled)
    np.testing.assert_allclose(unscaled, inp, atol=2e-04, rtol=2e-04)


@given(params_as_numpy_arrays())
def test_min_max_as_numpy_arrays(strat):
    _test_MnMxScaler(strat)


@given(valid_min_max_numpy_inp())
def test_nparrays(strat):
    _test_MnMxScaler(strat)


@given(valid_min_max_tensor_inp())
def test_tensors(strat):
    _test_MnMxScaler(strat)


@composite
def max_greater_than_min(draw):
    floatvals = floats(min_value=-100, max_value=100)
    a_min = draw(floatvals)
    a_max = draw(floats(min_value=a_min - 100, max_value=a_min))

    b_min = draw(floatvals)
    b_max = draw(floats(min_value=b_min - 100, max_value=b_min))

    scaleto = (a_min, a_max)
    scalefrom = (b_min, b_max)

    inp = draw(float_tensors(shape=array_shapes(min_dims=1, max_dims=3, min_side=2, max_side=10),
                             unique=True, elements=floats(min_value=b_min, max_value=b_max)))
    return scaleto, scalefrom


@given(tuples(floats(), floats()), tuples(floats(), floats()))
def test_MnMxScaler_init(scaleto, scalefrom):
    amin, amax = scaleto
    bmin, bmax = scalefrom
    if (amax <= amin) or (bmax <= bmin) or not all(map(np.isfinite, (amin, amax, bmin, bmax))):
        with pytest.raises(ValueError):
            scaler = MinMaxScaler(scaleto, scalefrom)
    else:
        scaler = MinMaxScaler(scaleto, scalefrom)
        assert scaler.scale_range == scaleto
        assert scaler.input_range == scalefrom
