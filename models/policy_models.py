from torch import nn
from torch.distributions import Normal

from ppo_pytorch.distributions import TruncatedNormal





class GaussianPolicy(nn.Module):

    def __init__(self, input_dim, hidden_size, output_dim, fixed_std=.5, hidden_act=nn.Tanh, output_act=nn.Tanh):
        super().__init__()
        self.std = fixed_std
        self.w1 = nn.Linear(input_dim, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.mean_head = nn.Linear(hidden_size, output_dim)
        self.hidden_act = hidden_act
        self.output_act = output_act


    def dist(self, x):
        return Normal(x, self.std)


    def forward(self, x):
        x = self.hidden_act()(self.w1(x))
        x = self.hidden_act()(self.w2(x))
        mean = self.output_act()(self.mean_head(x))
        return mean

    def sample_action(self, states):
        mean = self.forward(states)
        dist = self.dist(mean, self.std)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob

    def evaluate_action(self, states, actions):
        mean = self.forward(states)
        dist = self.dist(mean, self.std)
        logprob = dist.log_prob(actions)
        return logprob



class TruncatedGaussianPolicy(GaussianPolicy):

    def __init__(self, action_bounds=(-1., 1.), *args, **kwargs):
        super(TruncatedGaussianPolicy, self).__init__(*args, **kwargs)
        self.action_bounds = action_bounds

    def dist(self, x):
        return TruncatedNormal(x, self.std, *self.action_bounds)


    
    
