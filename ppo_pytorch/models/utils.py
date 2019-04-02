import functools
from torch import nn


def inspect_output_dim(model: nn.Module) -> tuple:
    return list(model.state_dict().items())[-1][-1].shape[-1]


def inspect_input_dim(model: nn.Module) -> tuple:
    return list(model.state_dict().items())[0][-1].shape[-1]


def reshape_output(*shape):
    """decorator will wrap *all* output tensors of the decorated
    method with a view of the provided shape...
    
    Args:
        *shape: tuple... valid shape for output tensors

    Example:

        ```Python

        @reshape_output(1, -1)
        def single_ouput_func():
            return torch.ones(10)

        single_output_func(5).shape
        out: (1, 10)


        @reshape_output(1, -1)
        def two_ouput_func():
            return torch.ones(10), torch.ones(5)

        a, b = two_ouput_func()
        a.shape
        out: (1, 10)

        b.shape
        out: (1, 5)

    """
    def expand_inner(f):
        @functools.wraps(f)
        def inner(*args, **kwargs):
            res = f(*args, **kwargs)
            if type(res) is tuple:
                return tuple(t.view(*shape) for t in res)
            return res.view(*shape)
        return inner
    return expand_inner
