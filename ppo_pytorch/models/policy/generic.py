import torch
from ppo_pytorch.models.network_head import NetworkHead
from ppo_pytorch.utils import classproperty


class PolicyHead(NetworkHead):

    """Generic Policy head : base class for all policy heads

    Attributes:
        discrete : if true the policy is discrete if false it's continuious
        Distribution : the distribution type for this policy
    """

    Distribution = NotImplemented
    discrete = NotImplemented

    @classproperty
    def continuous(cls):
        return not cls.discrete

    @classproperty
    def bounds(cls):
        try:
            return (cls.Distribution.support.lower_bound, cls.Distribution.support.upper_bound)
        except AttributeError:
            return None

    @classproperty
    def bounded_support(cls):
        return cls.bounds is not None


    @property
    def info(self):
        return dict(**super().info,
                    distribution = self.Distribution.__name__, 
                    action_space = 'discrete' if self.discrete else 'continuous', 
                    bounds = self.bounds)
  

    def _get_distribution(self, *args, **kwargs):
        return self.Distribution(*args, **kwargs)

    def distribution(self, x: torch.Tensor):
        x = self(x)
        return self._get_distribution(x)

    @torch.no_grad()
    def sample(self, state: torch.Tensor):
        dist = self.distribution(state)
        action = dist.sample()
        return action, dist.log_prob(action)

    def clipped_advantage_error(self, x: torch.Tensor, action: torch.Tensor, advantage: torch.Tensor, old_logprob: torch.Tensor, clip=.2, entropy_bonus=0.) -> torch.Tensor:

        dist = self.distribution(x)
        logprob = dist.log_prob(action)
        ratio = torch.exp(logprob - old_logprob)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage
        err = -torch.min(surr1, surr2).mean()
        return err - dist.entropy().mean() * self.entropy_bonus
