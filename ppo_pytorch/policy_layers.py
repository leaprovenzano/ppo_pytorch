from torch import nn
import torch
from torch.distributions import Normal, Beta, Categorical
from ppo_pytorch.distributions import UnimodalBeta, ScaledUnimodalBeta


class PolicyLayer(nn.Module):

    Distribution = NotImplemented
    discrete = NotImplemented

    def __init__(self, input_dims, action_dims):
        super().__init__()
        self.input_dims = input_dims
        self.action_dims = action_dims

    @property
    def out_features(self):
        return self.action_dims

    def get_distribution(self, *dist_params, **kwargs):
        return self.Distribution(*dist_params)

    def get_dist_params(self, x):
        raise NotImplementedError()

    def forward(self, x):
        return self.get_distribution(*self.get_dist_params(x))

    @torch.no_grad()    
    def sample(self, x):
        dist = self(x)
        samp = dist.sample()
        logprob = dist.log_prob(samp)
        return samp, logprob


class ContinuousPolicyLayer(PolicyLayer):

    discrete = False


class BetaPolicyLayer(ContinuousPolicyLayer):

    Distribution = UnimodalBeta
    bounds = (0, 1)

    def __init__(self, input_dims, action_dims, action_range=(0, 1)):
        super().__init__(input_dims, action_dims)
        if action_range != self.bounds:
            self.Distribution = ScaledUnimodalBeta.from_range(action_range)
            self.action_range = action_range

        self.activation = nn.Softplus()

        self.alpha = nn.Linear(self.input_dims, self.action_dims)
        self.beta = nn.Linear(self.input_dims, self.action_dims)

    def get_dist_params(self, x):
        a, b = self.alpha(x), self.beta(x)
        return self.activation(a), self.activation(b)

    def get_distribution(self, a, b):
        return self.Distribution(a, b)


class DiscretePolicyLayer(PolicyLayer):

    discrete = True


class CategoricalPolicyLayer(DiscretePolicyLayer):

    Distribution = Categorical

    def __init__(self, input_dims, action_dims, use_logits=True):
        super().__init__(input_dims, action_dims)
        self.linear = nn.Linear(self.input_dims, self.action_dims)

        self.uses_logits = use_logits
        if self.uses_logits:
            self.activation = nn.LogSoftmax(dim=-1)
            self._get_distribution = lambda x: self.Distribution(logits=x)
        else:
            self.activation = nn.Softmax(dim=-1)
            self._get_distribution = lambda x: self.Distribution(probs=x)

    def get_distribution(self, x):
        return self._get_distribution(x)

    def get_dist_params(self, x):
        x = self.linear(x)
        return (self.activation(x),)

    @torch.no_grad()    
    def sample(self, x):
        dist = self(x)
        samp = dist.sample()
        logprob = dist.log_prob(samp)
        return samp.unsqueeze(-1), logprob.unsqueeze(-1)
