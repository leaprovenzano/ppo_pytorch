import torch
import numpy as np


class RewardProcessor(object):

    def __init__(self, gamma=.99, positive_factor=2., clamp=(-1, 1)):
        self.gamma=gamma
        self.clamp = clamp
        self.positive_factor = positive_factor



    def shape(self, rewards):
        if self.positive_factor:
            rewards[rewards > 0] *=self.positive_factor
        if self.clamp:
            rewards = torch.clamp(rewards, *self.clamp)
        return rewards


    def compute_discount_returns(self, rewards):
        returns = torch.zeros_like(rewards)
        returns[-1] = rewards[-1]
        for i in reversed(range(rewards.size(0)-1)):
            returns[i] = returns[i+1]*self.gamma + rewards[i]
        return returns
        


class TrainHistory(object):

    def __init__(self):
        self.policy_loss = []
        self.value_loss = []

    def __len__(self):
        len(self.policy_loss)

    def collect_entry_stats(self, losses):
        stats = {'epochs': len(losses) if losses else 0}
        stats['min'] = np.min(losses) if losses else 0
        stats['max'] = np.max(losses) if losses else 0
        stats['mean'] = np.mean(losses) if losses else 0
        stats['std'] = np.std(losses) if losses else 0
        stats['last'] = losses[-1] if losses else 0
        return stats

    def update(self, attr, losses):
        entry = self.collect_entry_stats(losses)
        attr.append(entry)

    def update_policy_losses(self, losses):
        self.update(self.policy_loss, losses)

    def update_value_losses(self, losses):
        self.update(self.value_loss, losses)

    @property
    def last(self):
        return self.policy_loss[-1], self.value_loss[-1]

    def sub_format(self, d):
        formatter = '{}: {:.3f}'
        return ' '.join([formatter.format(k, v) for k, v in d.items() if k != 'epochs'])

    def __str__(self):
        formatter = '{} loss : {}'
        p_str = formatter.format(
            'policy', self.sub_format(self.policy_loss[-1]))
        v_str = formatter.format('value', self.sub_format(self.value_loss[-1]))
        return '{} | {}'.format(p_str, v_str)

