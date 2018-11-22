import torch
from torch import nn

from torch.distributions import Normal, Categorical

import numpy as np

from ppo_pytorch.distributions import TruncatedNormal
from ppo_pytorch.models.shared import NetworkHead


class PolicyLayer(NetworkHead):

    """base class for PolicyLayers which act as head for shared actor-critic or 
    standalone poicy models

    Attributes:
        action_layer (torch.nn.Linear): linear output layer of size output dim 
        activation (torch nn Activation | other callable class): Activation on action layer
                if none will just be an passthrough lambda
        input_dim (int): input nodes , note that the incoming features are assumed
                to be flat before getting here.
        output_dim (int): size of output ie the size of the action space
    """

    def __init__(self, *args, **kwargs):
        super(PolicyLayer, self).__init__(*args, **kwargs)

    def dist(self, x):
        raise NotImplementedError('the dist method for {} has not been implemented!'.format(self))

    def sample(self, x):
        dist = self.dist(x)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate_action(self, x, actions):
        dist = self.dist(x)
        logprob = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return logprob, entropy


class GaussianPolicy(PolicyLayer):

    def __init__(self, init_std=1., log_std_lr=1e-7, *args, **kwargs):
        super(GaussianPolicy, self).__init__(*args, **kwargs)
        self.log_std = nn.Parameter(torch.ones(1) * np.log(init_std))
        self.log_std_lr = log_std_lr

    @property
    def std(self):
        return torch.exp(self.log_std)

    def dist(self, x):
        x = self(x)
        return Normal(x, self.std.expand_as(x))

    def get_optimizer_parameters(self):
        log_std_p_dict = {'params': self.log_std, 'lr': self.log_std_lr}
        p_dict = {'params': map(lambda p: p[1], filter(lambda p: p[0] != 'log_std', self.named_parameters()))}
        if self.lr is not None:
            p_dict['lr'] = self.lr
        return [log_std_p_dict, p_dict]


class FixedGaussianPolicy(PolicyLayer):

    def __init__(self, fixed_std=.5, *args, **kwargs):
        super(FixedGaussianPolicy, self).__init__(*args, **kwargs)
        self.std = fixed_std

    def dist(self, x):
        x = self(x)
        return Normal(x, self.std)


class FixedTruncatedGaussianPolicy(FixedGaussianPolicy):

    def __init__(self, action_bounds=(-1., 1.), *args, **kwargs):
        super(FixedTruncatedGaussianPolicy, self).__init__(*args, **kwargs)
        self.action_bounds = action_bounds

    def dist(self, x):
        x = self(x)
        return TruncatedNormal(x, self.std, *self.action_bounds)


class TruncatedGaussianPolicy(GaussianPolicy):

    def __init__(self, action_bounds=(-1., 1.), *args, **kwargs):
        super(TruncatedGaussianPolicy, self).__init__(*args, **kwargs)
        self.action_bounds = action_bounds

    def dist(self, x):
        x = self(x)
        return TruncatedNormal(x, self.std.expand_as(x), *self.action_bounds)


class CategoricalPolicy(PolicyLayer):

    def __init__(self, *args, **kwargs):
        super(CategoricalPolicy, self).__init__(*args, **kwargs)
        self.output_activation = nn.Softmax(dim=-1)

    def dist(self, x):
        x = self(x)
        return Categorical(probs=x)
