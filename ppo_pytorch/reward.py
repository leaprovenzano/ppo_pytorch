import torch
import numpy as np


def discount(x, bootstrap=0., gamma=.99):
    """basic discount value function with optional bootstrap value

    Args:
        x (torch.tensor): time ordered values to discount.
        bootstrap (float, optional): If provided this will be the 
            future value to be discounted at the last step, zero by
            default assumes a full episode has been provided.
        gamma (float, optional): future return discount factor

    Returns:
        torch.tensor: a torch tensor like the one provided
            at input containing discounted future estimated
            rewards or whatever.
    """
    y = torch.zeros_like(x)
    future = bootstrap
    for i in reversed(range(len(x))):
        future = x[i] + gamma * future
        y[i] += future
    return y


def masked_discount(x, non_terminals, bootstrap=0., gamma=.99):
    """basic discount value function with optional bootstrap value

    Args:
        x (torch.tensor): time ordered values to discount.
        bootstrap (float, optional): If provided this will be the 
            future value to be discounted at the last step, zero by
            default assumes a full episode has been provided.
        non_terminals (torch.tensor) mask where teminal states == 0 and non
            terminals == 1
        gamma (float, optional): future return discount factor

    Returns:
        torch.tensor: a torch tensor like the one provided
            at input containing discounted future estimated
            rewards or whatever.
    """
    y = torch.zeros_like(x)
    y[-1] = x[-1] + (gamma * bootstrap * non_terminals[-1])
    for i in reversed(range(len(x) - 1)):
        future = gamma * y[i + 1] * non_terminals[i]
        y[i] = x[i] + future
    return y


class RandomGammaRewardProcessor(object):

    def __init__(self, gamma_min=.95, gamma_max=.995):
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    @property
    def gamma(self):
        return np.random.uniform(self.gamma_min, self.gamma_max)

    def compute_discount_returns(self, rewards, mask, bootstrap=0.):
        return masked_discount(rewards, mask, bootstrap=bootstrap, gamma=self.gamma)
