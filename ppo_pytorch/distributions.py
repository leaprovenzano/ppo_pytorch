from functools import partial
from torch.distributions import Beta
from ppo_pytorch.utils.scalers import MinMaxScaler


class UnimodalBeta(Beta):

    """Adjusted Beta(α + 1, β + 1) which when used along with softplus activation ensures our Beta distribution is always unimodal.

        Examples:
        
            >>> import torch
            >>> from ppo_pytorch.distributions import UnimodalBeta
            >>>
            >>> a = torch.tensor([[2.93, 0.65, 0.84],
            ...                   [0.52, 3.39, 5.86]])
            >>> b = torch.tensor([[1.40, 0.06, 0.55],
            ...                   [6.45, 3.83, 1.16]])
            >>> dist = UnimodalBeta(a, b)
            >>> dist.concentration1
            tensor([[3.9300, 1.6500, 1.8400],
                    [1.5200, 4.3900, 6.8600]])
            >>> dist.concentration0
            tensor([[2.4000, 1.0600, 1.5500],
                    [7.4500, 4.8300, 2.1600]])
            >>> dist.sample()
            tensor([[0.4651, 0.3861, 0.5077],
                    [0.2856, 0.5835, 0.7325]])

    """

    def __init__(self, alpha, beta, validate_args=None):
        super().__init__(alpha + 1, beta + 1, validate_args=validate_args)


class ScaledUnimodalBeta(UnimodalBeta):

    """Unimodal beta with a scaler attatched that rescales at the following points:

       - outputs to the sample method are rescaled from [0, 1] to the specified output range
       - inputs to log_prob are rescaled from the specified output range back to [0,1]


        Examples:

            >>> import torch
            >>> from ppo_pytorch.distributions import ScaledUnimodalBeta
            >>> 
            >>> # make a new ScaledUniModalBeta factory from with your output range
            >>> ScaledDist = ScaledUnimodalBeta.from_range((-1, 1))
            >>> 
            >>> # that was one time setup, now you can use scaled dist as normal with alpha & beta params 
            >>> # and it will always scale your samples and make sure logprobs are correct. 
            >>> a = torch.tensor([[2.93], [0.65]])
            >>> b = torch.tensor([[1.40], [0.06]])
            >>> dist = ScaledDist(a, b)
            >>> dist.sample()
            tensor([[ 0.2011],
                    [-0.6362]])
            >>> dist.log_prob(torch.tensor([[ 0.2011], [-0.6362]]))
            tensor([[ 0.6600],
                    [-0.5400]])
    """

    @classmethod
    def from_range(cls, new_range):
        return partial(cls, new_range)

    def __init__(self, output_range, alpha, beta, validate_args=None):
        super().__init__(alpha, beta, validate_args=validate_args)
        self._sample_scaler = MinMaxScaler((0, 1), output_range)

    def sample(self, *args, **kwargs):
        samp = super().sample(*args, **kwargs)
        return self._sample_scaler.scale(samp)

    def log_prob(self, sample, *args, **kwargs):
        return super().log_prob(self._sample_scaler.inverse_scale(sample), *args, **kwargs)
