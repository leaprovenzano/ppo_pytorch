import numpy as np
import torch 


from typing import Tuple, Union

from ppo_pytorch.utils.descriptors import MinMaxRange


class MinMaxScaler(object):

    inrange = MinMaxRange()
    outrange = MinMaxRange()

    def __init__(self, inrange: Tuple[float, float], outrange: Tuple[float, float]):
        self.inrange = inrange
        self.outrange = outrange

    @staticmethod
    def standardize(x: Union[np.array, torch.tensor], rng: MinMaxRange) -> Union[np.array, torch.tensor]:
        return (x - rng.min) / rng.span

    def _scale(self, x: Union[np.array, torch.tensor], a: MinMaxRange, b: MinMaxRange) -> Union[np.array, torch.tensor]:
        return self.standardize(x, a) * b.span + b.min

    @torch.no_grad()
    def scale(self, x: Union[np.array, torch.tensor]):
        return self._scale(x, self.inrange, self.outrange)

    @torch.no_grad()
    def inverse_scale(self, x: Union[np.array, torch.tensor]):
        return self._scale(x, self.outrange, self.inrange)

    @torch.no_grad()
    def __call__(self, x: Union[np.array, torch.tensor]):
        return self.scale(x)
