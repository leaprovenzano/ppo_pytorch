import torch
import numpy as np


class RewardProcessor(object):

    def __init__(self, gamma=.99):
        self.gamma = gamma

    def _shape(self, rewards):
        return rewards

    def shape(self, rewards):
        return self._shape(rewards)

    def compute_discount_returns(self, rewards):
        returns = torch.zeros_like(rewards)
        returns[-1] = rewards[-1]
        for i in reversed(range(rewards.size(0) - 1)):
            returns[i] = returns[i + 1] * self.gamma + rewards[i]
        return returns


class ShapedRewardProcessor(RewardProcessor):

    def __init__(self, scale=1., clip=None, *args, **kwargs):
        super(ShapedRewardProcessor, self).__init__(*args, **kwargs)
        self.scale = scale
        self.clip = clip

    def _shape(self, rewards):
        rewards = rewards * self.scale
        if self.clip:
            rewards = torch.clamp(rewards, *self.clip)
        return rewards


class ScaledPositiveRewardProcessor(ShapedRewardProcessor):

    def __init__(self, positive_scale=2., *args, **kwargs):
        super(ScaledPositiveRewardProcessor, self).__init__(*args, **kwargs)
        self.positive_scale = positive_scale

    def shape(self, rewards):
        rewards[rewards > 0] *= self.positive_scale
        return super(ScaledPositiveRewardProcessor, self)._shape(rewards)


class RandomGammaRewardProcessor(RewardProcessor):

    def __init__(self, gamma_min=.95, gamma_max=.995):
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    @property
    def gamma(self):
        return np.random.uniform(self.gamma_min, self.gamma_max)

    def compute_discount_returns(self, rewards):
        returns = torch.zeros_like(rewards)
        returns[-1] = rewards[-1]
        gamma = self.gamma
        for i in reversed(range(rewards.size(0) - 1)):
            returns[i] = returns[i + 1] * gamma + rewards[i]
        return returns
