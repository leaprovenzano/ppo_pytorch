import numpy as np

import torch
from torch import nn
from torch.distributions import Normal, Beta

from .generic import PolicyHead
from ppo_pytorch.utils import MnMxScaler


class ContinuousPolicy(PolicyHead):

    """Summary

    Attributes:
        discrete (bool): Description

    """
    discrete = False


class BetaPolicy(ContinuousPolicy):

    """Summary

    Attributes:
        ab_activation (TYPE): Description
        alpha_layer (TYPE): Description
        beta_layer (TYPE): Description
    """

    Distribution = Beta

    def __init__(self, *args, **kwargs):
        super(BetaPolicy, self).__init__(*args, **kwargs)
        self.ab_activation = nn.Softplus()
        self.alpha_layer = nn.Linear(self._hidden_ouput_dim, self.output_dim)
        self.beta_layer = nn.Linear(self._hidden_ouput_dim, self.output_dim)

    def output_layer(self, x):
        a = self.alpha_layer(x)
        b = self.beta_layer(x)
        return a, b

    def output_activation(self, a, b):
        return self.ab_activation(a) + 1, self.ab_activation(b) + 1

    def forward(self, inp):
        x = self.hidden(inp)
        a, b = self.output_layer(x)
        a, b = self.output_activation(a, b)
        return a, b

    def distribution(self, x: torch.Tensor):
        x = self(x)
        return self.get_distribution(*x)


class ScaledBetaPolicy(BetaPolicy):

    """Summary

    Attributes:
        ab_activation (TYPE): Description
        alpha_layer (TYPE): Description
        beta_layer (TYPE): Description
    """

    def __init__(self, output_dim, hidden, action_scale=(-1, 1), *args, **kwargs):
        super(ScaledBetaPolicy, self).__init__(output_dim, hidden, *args, **kwargs)
        self.action_scaler = MnMxScaler(action_scale, self.bounds)

    @property
    def info(self):
        return dict(**super().info, output_action_range = self.action_scaler.range)

    def sample(self, state: torch.Tensor):
        action, logprob = super().sample(state)
        return self.action_scaler.scale(action), logprob

    def clipped_advantage_error(self, x: torch.Tensor, action: torch.Tensor, advantage: torch.Tensor, old_logprob: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        action_ = self.action_scaler.inverse_scale(action)
        return super().clipped_advantage_error(x, action_, advantage, old_logprob)


# class BaseGaussianPolicy(ContinuousPolicy):

#     """Base Class for gaussian policies, essentially this is functionally equivilant to
#     the fixed gaussian policy.
#     """

#     _min_init_std = .1
#     Distribution = Normal

#     @classmethod
#     def _check_std(cls, std):
#         if std < cls._min_init_std:
#             raise ValueError(f'{cls.__name__} must be initilized with a std >= {cls._min_init_std}')
#         pass

#     def __init__(self, std=.5, *args, **kwargs):
#         self._check_std(std)
#         super().__init__(*args, **kwargs)
#         self._std = std

#     def distribution(self, x):
#         return self.Distribution(x, self.std)

#     @property
#     def std(self):
#         return self._std


# class GaussianPolicy(BaseGaussianPolicy):

#     """Summary

#     Attributes:
#         log_std (TYPE): Description
#         std_lr (TYPE): Description
#     """

#     def __init__(self, std_lr=1e-6, *args, **kwargs):
#         super(GaussianPolicy, self).__init__(*args, **kwargs)
#         self.log_std = nn.Parameter(torch.ones(1) * np.log(self._std))
#         self.std_lr = std_lr

#     @property
#     def std(self):
#         return torch.exp(self.log_std)

#     def distribution(self, x):
#         return self.Distribution(x, self.std.expand_as(x))

#     def get_optimizer_parameters(self):
#         log_std_p_dict = {'params': self.log_std, 'lr': self.std_lr}
#         p_dict = {'params': map(lambda p: p[1], filter(lambda p: p[0] != 'log_std', self.named_parameters()))}
#         if self.lr is not None:
#             p_dict['lr'] = self.lr
#         return [log_std_p_dict, p_dict]


# FixedGaussianPolicy = BaseGaussianPolicy