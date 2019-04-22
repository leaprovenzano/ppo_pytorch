from functools import partial
from torch.distributions import Beta
from ppo_pytorch.utils.scalers import MinMaxScaler


class UnimodalBeta(Beta):

    """Adjusted Beta(α + 1, β + 1) which when used along with softplus activation ensures our Beta distribution is always unimodal.

    """

    def __init__(self, concentration1, concentration0, validate_args=None):
        super().__init__(concentration1 + 1, concentration0 + 1, validate_args=validate_args)


class ScaledUnimodalBeta(UnimodalBeta):

    """Summary

    Attributes:
        sample_scaler (TYPE): Description
    """

    @classmethod
    def from_range(cls, new_range):
        return partial(cls, new_range)

    def __init__(self, output_range, concentration1, concentration0, validate_args=None):
        super().__init__(concentration1, concentration0, validate_args=validate_args)
        self.sample_scaler = MinMaxScaler((0, 1), output_range)

    def sample(self, *args, **kwargs):
        samp = super().sample(*args, **kwargs)
        return self.sample_scaler.scale(samp)

    def log_prob(self, sample, *args, **kwargs):
        return super().log_prob(self.sample_scaler.inverse_scale(sample), *args, **kwargs)
