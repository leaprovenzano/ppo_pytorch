import torch
from torch import nn
from torch.distributions import Categorical

from .generic import PolicyHead
from ppo_pytorch.utils import reshape_outputs


class DiscretePolicy(PolicyHead):
    discrete = True


class CategoricalPolicy(DiscretePolicy):

    Distribution = Categorical

    def __init__(self, *args, **kwargs):
        super(DiscretePolicy, self).__init__(*args, **kwargs)
        self.output_activation = nn.LogSoftmax(dim=-1)

    def _get_distribution(self, x):
        return self.Distribution(logits=x)

    @reshape_outputs(1, -1)
    def sample(self, x):
        return super().sample(x)
