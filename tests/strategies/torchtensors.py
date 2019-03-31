import torch

from hypothesis.strategies import floats, integers, lists, sampled_from, just, one_of
from hypothesis.extra.numpy import arrays, array_shapes, integer_dtypes

float_type_strings = ['float32', 'float64']
int_type_strings = ['int', 'uint8', 'int32', 'int64']


float_dtypes = sampled_from(float_type_strings)
int_dtypes = sampled_from(int_type_strings)


float_type_strings = ['float32', 'float64']
int_type_strings = ['int', 'uint8', 'int32', 'int64']


def dypes(dtype_strings):
    return sampled_from(list(dtype_strings))


def tensors(dtypes, *args, **kwargs):
    try:
        dtypes.validate()
    except AttributeError:
        if isinstance(dtypes, str):
            return tensors(just(dtypes), *args, **kwargs)
        elif type(dtypes) in (list, tuple):
            return tensors(sampled_from())
        else:
            raise ValueError('dtype : {} not understood ... please provide a list or tuple of type strings or a stategy')
    arrs = arrays(dtypes, *args, **kwargs)
    return arrs.map(torch.as_tensor)


def float_tensors(dtypes=float_dtypes, *args, **kwargs):
    return tensors(dtypes=float_dtypes, *args, **kwargs)


def int_tensors(dtypes=int_dtypes, *args, **kwargs):
    return tensors(dtypes=dtypes, *args, **kwargs)

def variable_batch_shape(feature_shape, min_len=0, max_len=10000):
    var_shapes = integers(min_value=1, max_value=max_len).map(lambda x : tuple([x]))
    if min_len == 0:
        var_shapes = one_of(just(tuple([])), var_shapes)
    shapes = var_shapes.map(lambda x : x + feature_shape)

    return shapes