import pytest

import numpy as np

from hypothesis import given
from hypothesis.strategies import integers, floats, composite
from hypothesis.extra.numpy import arrays, array_shapes

from ppo_pytorch.utils.descriptors import MinMaxRange

basic_arrays = arrays(dtype='float',
                      shape=array_shapes(min_dims=1, max_dims=3, min_side=2, max_side=10),
                      elements=floats(min_value=-1000, max_value=1000))


class DummyDescriptorHolder(object):

    rng = MinMaxRange()

    def __init__(self, low, high):
        self.rng = (low, high)


@composite
def valid_high_low_arrays(draw, epsilon):
    low = draw(basic_arrays)
    high = low + draw(arrays(dtype='float', shape=low.shape,
                             elements=floats(min_value=.001, max_value=100)))
    return low, high


@composite
def non_matching_shape_arrays(draw, epsilon):
    low = draw(basic_arrays)
    highshape = array_shapes(min_dims=1, max_dims=3, min_side=2, max_side=10).filter(lambda x: x != low.shape)
    high = draw(arrays(dtype='float', shape=highshape,
                       elements=floats(min_value=low.max() + epsilon, max_value=low.max() + 1)))
    return low, high


@composite
def low_gte_high(draw, inputs):
    low = draw(inputs)
    sub = floats(min_value=0, max_value=100)
    high = low - draw(sub)
    return low, high


class TestMinMaxRange(object):

    def assert_raises_value_error(self, low, high):
        with pytest.raises(ValueError):
            DummyDescriptorHolder(low, high)

    def assert_valid(self, low, high):
        holder = DummyDescriptorHolder(low, high)
        assert np.all(holder.rng.low == low)
        assert np.all(holder.rng.high == high)
        assert np.all(holder.rng.span == high - low)

    @given(valid_high_low_arrays(MinMaxRange.epsilon))
    def test_valid_arrays_pass(self, rng):
        low, high = rng
        self.assert_valid(low, high)

    @given(arrays(dtype='float',
                  shape=array_shapes(min_dims=1, max_dims=3, min_side=2, max_side=10),
                  elements=floats(min_value=-1000, max_value=1000)))
    def test_arr_and_scaler(self, arr):
        self.assert_valid(arr, arr.max() + 1)
        self.assert_valid(arr.min() - 1, arr)

    @given(non_matching_shape_arrays(MinMaxRange.epsilon))
    def test_invalid_array_shapes(self, rng):
        low, high = rng
        self.assert_raises_value_error(low, high)

    @given(low_gte_high(
        (floats(min_value=-100, max_value=100, width=32) | integers(min_value=-100, max_value=100) | basic_arrays)))
    def test_low_values_gte_high_are_invalidated(self, rng):
        low, high = rng
        self.assert_raises_value_error(low, high)
#