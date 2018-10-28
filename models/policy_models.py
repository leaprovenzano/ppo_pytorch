from torch import nn
from torch.distributions import Normal

from ppo_pytorch.distributions import TruncatedNormal
from ppo_pytorch.models.base import SimpleMLP




class GaussianPolicy(SimpleMLP):

    def __init__(self, input_dim, hidden_dims, output_dim, hidden_act=nn.Tanh, output_act=nn.Tanh, fixed_std=.5):
        super(GaussianPolicy, self).__init__( input_dim, hidden_dims,output_dim, hidden_act=nn.Tanh, output_act=nn.Tanh)
        self.std = fixed_std


    def dist(self, x):
        return Normal(x, self.std)


    def sample_action(self, states):
        mean = self.forward(states)
        dist = self.dist(mean)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate_action(self, states, actions):
        mean = self.forward(states)
        dist = self.dist(mean)
        logprob = dist.log_prob(actions)
        return logprob



class TruncatedGaussianPolicy(GaussianPolicy):

    def __init__(self, action_bounds=(-1., 1.), *args, **kwargs):
        super(TruncatedGaussianPolicy, self).__init__(*args, **kwargs)
        self.action_bounds = action_bounds

    def dist(self, x):
        return TruncatedNormal(x, self.std, *self.action_bounds)


    
    
