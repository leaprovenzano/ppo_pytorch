import numpy as np
import torch


from typing import Tuple, Union

from ppo_pytorch.utils.descriptors import ValidationDescriptor


class MinMaxRange(ValidationDescriptor):

    """Open pseudo Descriptor for range objects... pseudo because this descriptors attributes can be accessed directly once bound.

    MinMaxRange is parameterized by `min` and `max` which may be numpy arrays, torch tensors, floats or ints. Validation is preformed
    on min and max to be sure the values confirm to certain constraints:

        - both min and max must contain only finite values
        - min must be < max by at least epsilon
        - if min and max are arrays or matrices they must have the same shape
        - if only one of min and max are arraylike the other value may be a scaler

    Attributes:
        epsilon (float): minimum span between high and low. 1e-4 by default.
        high (float | np.array | torch.Tensor): high end of range
        low (float | np.array | torch.Tensor): low end of range

    """

    epsilon = 1e-4

    def __init__(self, low=None, high=None):
        self.low = low
        self.high = high

    @property
    def value(self):
        return self.low, self.high

    @property
    def span(self):
        return abs(self.high - self.low)

    @property
    def min(self):
        return self.low

    @property
    def max(self):
        return self.high

    def __set__(self, instance, rng):
        self._validate(*rng)
        self.low, self.high = rng

    def __get__(self, instance, owner):
        return self

    def __repr__(self):
        return f'{self.__class__.__name__} : ({self.low}, {self.high})'

    def __eq__(self, other):
        return self.low == other.low and self.high == other.high

    def _validate_values(self, values, attrname):
        if not np.all(np.isfinite(np.asarray(values))):
            msg = f'values {self.name}.{attrname} where are not finite (and are therefore invalid)'
            self._raise_on_invalidation(msg, values)
        return True

    def _type_and_dims_validate(self, low, high):

        if isinstance(low, type(high)):
            if isinstance(low, (np.ndarray, torch.Tensor)):
                if not low.shape == high.shape:
                    self._raise_on_invalidation('low and high values must be of the same shape',
                                                f'low= {low.shape}, high= {high.shape}')

        elif isinstance(low, (np.ndarray, torch.Tensor)):
            if not isinstance(high, (float, int)):
                self._raise_on_invalidation(f'if low is of type {type(low)} high must be of same type and shape OR a scaler')
            elif isinstance(high, (np.ndarray, torch.Tensor)):
                if not isinstance(low, (float, int)):
                    self._raise_on_invalidation(f'if high is of type {type(high)} low must be of same type and shape OR a scaler')

    def _validate(self, low, high):
        self._validate_values(low, 'low')
        self._validate_values(high, 'high')

        # if check dims should this be an array or tensor
        self._type_and_dims_validate(low, high)

        if np.any(np.asarray(high - low) < self.epsilon):
            self._raise_on_invalidation(f'low values must be less than high values with a margin of at least {self.epsilon}', f'low= {low}  high= {high}')

        return True


class MinMaxScaler(object):

    """Min max scaler built with MinMaxRange descriptiors. MinMaxScaler is instantiated
    with two tuples, inrange and outrange which may be floats, or matching shaped numpy arrays 
    or torch.Tensors or some mix of scalar float and array. MinMaxRange will validate these inputs
    and error where they are inappropriate

    Examples:

        instantiating a MinMaxScaler with inrange and outrange will give you an instance with two
        attributes `inrange` and `outrange` which are validated instances of `MinMaxRange` and interface
        methods `scale` and `inverse_scale`:

        >>> from ppo_pytorch.utils import MinMaxScaler
        >>>
        >>> scaler = MinMaxScaler((0., 1.), (-1., 1.))
        >>> print(scaler.inrange, scaler.outrange)
        (MinMaxRange : (0.0, 1.0), MinMaxRange : (-1.0, 1.0))


        Instantiation with numpy arrays also works just fine:

        >>> import numpy as np
        >>>
        >>> arrscaler = MinMaxScaler((np.zeros(3), np.ones(3)), (np.ones(3)*-1., np.ones(3)))
        >>> arrscaler.inrange, arrscaler.outrange
        (MinMaxRange : ([0. 0. 0.], [1. 1. 1.]), MinMaxRange : ([-1. -1. -1.], [1. 1. 1.]))

        so long as the array shape matches what you're trying to scale everything will be ok:

        >>> arrscaler.scale(np.random.uniform(size=(2, 3)))
        array([[ 0.55849497,  0.56310639,  0.95245284],
               [-0.00529875,  0.13534943, -0.61108798]])


        arrays  and floats can also be mixed:

        >>> scaler = MinMaxScaler((np.zeros(3), np.ones(3)), (-1., 1))
        >>> scaler.inrange, scaler.outrange
        (MinMaxRange : ([0. 0. 0.], [1. 1. 1.]), MinMaxRange : (-1.0, 1))

        torch Tensors also work just fine:

        >>> import torch
        >>>
        >>> tensorscaler =  MinMaxScaler((torch.zeros(3), torch.ones(3)), (torch.ones(3)*-1., torch.ones(3)))
        >>> tensorscaler.scale(torch.rand(2, 3))
        tensor([[ 0.5409,  0.0543,  0.8521],
                [-0.5484,  0.7508, -0.9703]])

    Attributes:
        inrange (MinMaxRange):  MinMaxRange for input
        outrange (MinMaxRange): MinMaxRange for  output

    Args:

        inrange : input data range ((float | np.array | torch.Tensor), (float | np.array | torch.Tensor))
        outrange : output data range ((float | np.array | torch.Tensor), (float | np.array | torch.Tensor))

    """

    inrange = MinMaxRange()
    outrange = MinMaxRange()

    def __init__(self, inrange: Tuple[float, float], outrange: Tuple[float, float]):
        self.inrange = inrange
        self.outrange = outrange

    @staticmethod
    def _standardize(x: Union[np.array, torch.tensor], rng: MinMaxRange) -> Union[np.array, torch.tensor]:
        return (x - rng.min) / rng.span

    def _scale(self, x: Union[np.array, torch.tensor], a: MinMaxRange, b: MinMaxRange) -> Union[np.array, torch.tensor]:
        return self._standardize(x, a) * b.span + b.min

    @torch.no_grad()
    def scale(self, x: Union[np.array, torch.tensor]):
        """scale input from range specified in `inrange` to `outrange`.

        Args:
            x (Union[np.array, torch.tensor]): input should be in `inrange`

        Example:

        >>> import torch
        >>>
        >>> scaler =  MinMaxScaler((torch.zeros(3), torch.ones(3)),
        ...                       (torch.ones(3)*-1., torch.ones(3)))
        >>>
        >>> inp = torch.Tensor([[.1, .2, .3],
        ...                     [.4, .5, .6]])
        >>>
        >>> scaler.scale(inp)
        tensor([[-0.8000, -0.6000, -0.4000],
                [-0.2000,  0.0000,  0.2000]])

        Notes:

            - where inrange or outrange where parameterized with tensors or arrays inputs should share the last dimension
            - this method is `torch.no_grad` decorated and so does not track gradient on this operation.


        """
        return self._scale(x, self.inrange, self.outrange)

    @torch.no_grad()
    def inverse_scale(self, x: Union[np.array, torch.tensor]):
        """inverse scale a transformed input back to the original range.

        Args:
            x (Union[np.array, torch.tensor]): input, usually output of previously scaled data.


        Example:

            >>> import torch
            >>>
            >>> scaler =  MinMaxScaler((torch.zeros(3), torch.ones(3)),
            ...                       (torch.ones(3)*-1., torch.ones(3)))
            >>>
            >>> inp = torch.Tensor([[.1, .2, .3],
            ...                     [.4, .5, .6]])
            >>>
            >>> scaled = scaler.scale(inp)
            >>> scaler.inverse_scale(scaled)
            tensor([[0.1000, 0.2000, 0.3000],
                    [0.4000, 0.5000, 0.6000]])

        Notes:

            - where inrange or outrange where parameterized with tensors or arrays inputs should share the last dimension
            - this method is `torch.no_grad` decorated and so does not track gradient on this operation.

        """
        return self._scale(x, self.outrange, self.inrange)

    @torch.no_grad()
    def __call__(self, x: Union[np.array, torch.tensor]):
        return self.scale(x)
