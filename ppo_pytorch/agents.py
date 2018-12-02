import numpy as np

import torch
from torch import nn

from ppo_pytorch.utils import TrainHistory
from torch.autograd import Variable


class PPOAgent(object):

    @classmethod
    def from_saved(cls, path, env_manager, model, optimizer):
        cp = torch.load(path)
        agent = cls(env_manager, model, optimizer)
        for k, v in cp['attrs'].items():
            setattr(agent, k, v)

        for k, v in cp['state_dicts'].items():
            getattr(agent, k).load_state_dict(v)
        return agent

    def __init__(self, env_manager, model, optimizer, epochs=20, grad_norm=0.5, normalize_advantage=True, clip_epsilon=.2, value_loss_coef=.5, entropy_bonus_coef=0.):
        self.value_loss_coef = value_loss_coef
        self.entropy_bonus_coef = entropy_bonus_coef
        self.envs = env_manager
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.grad_norm = grad_norm
        self.normalize_advantage = normalize_advantage
        self.clip_epsilon = clip_epsilon
        self.episode_rewards = []
        self.entropy_log = []
        self.episode = 0

    @property
    def params(self):
        return {k: v for k, v in self.__dict__.items() if not hasattr(v, '__dict__')}

    def episode_logstr(self, n, loss, policy_loss, value_loss, entropy):
        eps_logstr = 'episode {}, reward avg {:.2f} | samples: {} --  loss: {:.4f} | policy loss: {:.4f} |value loss: {:.4f} | entropy: {:.4f}'
        return eps_logstr.format(self.episode, self.episode_rewards[-1],
                                 n, loss, policy_loss, value_loss, entropy)

    def run_episode(self, render=False, reset=True):
        if reset:
            self.envs.reset()
        while not self.envs.ready:
            with torch.no_grad():
                actions, logprobs, values = self.model.sample_action(self.envs.current_states)
            self.envs.step(actions, logprobs, values)

        self.episode_rewards.append(self.envs.mean_reward)
        n, loss, policy_loss, value_loss, entropy = self.update()
        print(self.episode_logstr(n, loss, policy_loss, value_loss, entropy))
        self.episode += 1

    def estimate_advantage(self, returns, values):
        advantages = Variable(returns).view(-1, 1) - Variable(values.data)
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        return advantages.squeeze()

    def policy_loss(self, log_probs, old_log_probs, advantages):
        ratio = torch.exp(log_probs - Variable(old_log_probs.data))
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        return -torch.min(surr1, surr2).mean()

    def value_loss(self, y_true, y_pred):
        return (0.5 * (y_true.view(-1, 1) - y_pred.view(-1, 1)).pow(2)).mean()

    def update(self):
        states, actions, returns, old_logprobs, pred_values = self.envs.get_train_data()

        states = Variable(states)
        actions = Variable(actions)
        returns = Variable(returns)
        old_logprobs = Variable(old_logprobs.data)

        policy_losses = []
        value_losses = []
        losses = []
        entropies = []

        advantages = self.estimate_advantage(returns, pred_values)

        for i in range(self.epochs):
            log_probs, entropy, values = self.model.evaluate_actions(states, actions)
            policy_loss = self.policy_loss(log_probs, old_logprobs, advantages)
            value_loss = self.value_loss(returns, values)

            self.optimizer.zero_grad()
            loss = (value_loss * self.value_loss_coef + policy_loss - entropy * self.entropy_bonus_coef)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.grad_norm)
            self.optimizer.step()
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            losses.append(loss.item())
            entropies.append(entropy.item())
        entropy_mean = np.mean(entropies)
        self.entropy_log.append(entropy_mean)
        return len(states), losses[-1], np.sum(policy_losses), value_losses[-1], entropy_mean

    def save(self, path):
        out = {'attrs': self.params}
        state_dicts = {}

        state_dicts['model'] = self.model.state_dict()
        state_dicts['optimizer'] = self.optimizer.state_dict()
        out['state_dicts'] = state_dicts
        torch.save(out, path)

    def run_test_episode(self, env, render=True, sample=False, ):
        discrete = env.action_space.dtype in [int, np.int64, np.int32]
        reward = 0
        state = env.reset()
        done = False
        while not done:

            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                if sample:
                    action, _, _ = self.model.sample_action(state)

                else:
                    action, _ = self.model(state)

                    if discrete:
                        action = np.asarray(action)[0].argmax()
                    else:

                        action = np.clip(np.asarray(action)[0], env.action_space.low, env.action_space.high)

            state, r, done, _ = env.step(action)
            reward += r
            env.render()

        return reward
