import functools
from torch import nn

def inspect_output_dim(model: nn.Module) -> tuple:
    return list(model.state_dict().items())[-1][-1].shape[-1]


def inspect_input_dim(model: nn.Module) -> tuple:
    return list(model.state_dict().items())[0][-1].shape[-1]


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
