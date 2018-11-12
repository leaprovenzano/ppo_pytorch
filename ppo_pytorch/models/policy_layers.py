import torch
from torch import nn

from torch.distributions import Normal, Categorical

import numpy as np

from ppo_pytorch.distributions import TruncatedNormal


class PolicyLayer(nn.Module):

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

    def __init__(self, input_dim, output_dim, activation=None):
        super(PolicyLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.action_layer = nn.Linear(self.input_dim, self.output_dim)
        self.activation = activation() if activation else lambda x: x

    def forward(self, x):
        x = self.action_layer(x)
        x = self.activation(x)
        return x

    def dist(self, x):
        raise NotImplementedError('the dist method for {} has not been implemented!'.format(self))

    def sample(self, x):
        dist = self.dist(x)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate_action(self, x, actions):
        dist = self.dist(x)
        logprob = dist.log_prob(actions)
        return logprob


class GaussianPolicy(PolicyLayer):

    def __init__(self, input_dim, output_dim, activation=nn.Tanh, init_std=.5):
        super(GaussianPolicy, self).__init__(input_dim, output_dim, activation=activation)
        self.log_std = nn.Parameter(torch.ones(1) * np.log(init_std))

    @property
    def std(self):
        return torch.exp(self.log_std)

    def dist(self, x):
        x = self(x)
        return Normal(x, self.std.expand_as(x))


class FixedGaussianPolicy(PolicyLayer):

    def __init__(self, input_dim, output_dim, activation=nn.Tanh, fixed_std=.5):
        super(FixedGaussianPolicy, self).__init__(input_dim, output_dim, activation=activation)
        self.std = fixed_std

    def dist(self, x):
        x = self(x)
        return Normal(x, self.std)


class FixedTruncatedGaussianPolicy(FixedGaussianPolicy):

    def __init__(self, action_bounds=(-1., 1.), *args, **kwargs):
        super(FixedTruncatedGaussianPolicy, self).__init__(*args, **kwargs)
        self.action_bounds = action_bounds

    def dist(self, x):
        return TruncatedNormal(x, self.std, *self.action_bounds)


class TruncatedGaussianPolicy(GaussianPolicy):

    def __init__(self, action_bounds=(-1., 1.), *args, **kwargs):
        super(TruncatedGaussianPolicy, self).__init__(*args, **kwargs)
        self.action_bounds = action_bounds

    def dist(self, x):
        return TruncatedNormal(x, self.std.expand_as(x), *self.action_bounds)


class CategoricalPolicy(PolicyLayer):

    def __init__(self, input_dim, output_dims, activation=nn.Softmax):
        super(CategoricalPolicy, self).__init__(input_dim, output_dims, output_activation=activation)

    def dist(self, x):
        return Categorical(x)
