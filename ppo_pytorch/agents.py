import numpy as np

import torch
from torch import nn

from ppo_pytorch.utils import TrainHistory
from torch.autograd import Variable


class PPOAgent(object):

    @classmethod
    def from_saved(cls, path, env_manager, policy_model, value_model, policy_optimizer, value_optimizer):
        cp = torch.load(path)
        agent = cls(env_manager, policy_model, value_model,
                    policy_optimizer, value_optimizer)
        for k, v in cp['attrs'].items():
            setattr(agent, k, v)

        for k, v in cp['state_dicts'].items():
            getattr(agent, k).load_state_dict(v)
        return agent

    def __init__(self, env_manager, policy_model, value_model, policy_optimizer, value_optimizer, policy_epochs=20, value_epochs=20,
                 policy_grad_norm=0.5, value_grad_norm=0.5, normalize_advantage=True, clip_epsilon=.2):

        self.envs = env_manager
        self.policy_model = policy_model
        self.value_model = value_model
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer
        self.policy_epochs = policy_epochs
        self.value_epochs = value_epochs
        self.policy_grad_norm = policy_grad_norm
        self.value_grad_norm = value_grad_norm
        self.normalize_advantage = normalize_advantage
        self.clip_epsilon = clip_epsilon
        self.episode_rewards = []
        self.episode = 0
        self.history = TrainHistory()

    @property
    def params(self):
        return {k: v for k, v in self.__dict__.items() if not hasattr(v, '__dict__')}

    @property
    def policy_std(self):
        if type(self.policy_model.std) is float:
            return self.policy_model.std
        return np.exp(self.policy_model.std.detach().numpy()[0])

    def episode_logstr(self, train_stats):
        eps_logstr = 'episode {}, reward avg {:.2f} | {} |  policy std : {:.4f}'
        return eps_logstr.format(self.episode, self.episode_rewards[-1],
                                 train_stats,
                                 self.policy_std)

    def run_episode(self, render=False):
        self.envs.reset()
        while not self.envs.ready:
            with torch.no_grad():
                actions, logprobs = self.policy_model.sample_action(self.envs.current_states)
            self.envs.step(actions, logprobs)

        self.episode_rewards.append(self.envs.mean_reward)
        update_stats = self.train()
        print(self.episode_logstr(update_stats))
        self.episode += 1

    def train(self):
        _, nsamples = self.update()
        return 'nsamples : {}'.format(nsamples)

    def estimate_advantage(self, returns, states):
        advantages = Variable(returns).view(-1, 1) - Variable(self.value_model.predict(states).data).view(-1, 1)
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        return advantages

    def policy_loss(self, states, actions, old_log_probs, advantages):
        log_probs = self.policy_model.evaluate_action(states, actions)
        ratio = torch.exp(log_probs - Variable(old_log_probs.data))
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        return -torch.min(surr1, surr2).mean()

    def value_loss(self, y_true, y_pred):
        return (0.5 * (y_true.view(-1, 1) - y_pred.view(-1, 1)).pow(2)).mean()

    def policy_step(self, states, actions, old_log_probs, advantages):
        loss = self.policy_loss(states, actions, old_log_probs, advantages)
        self.policy_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.policy_grad_norm)
        self.policy_optimizer.step()
        return loss.item()

    def value_step(self, states, returns):
        value_pred = self.value_model.predict(states)
        value_loss = self.value_loss(returns, value_pred)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value_model.parameters(), self.value_grad_norm)
        self.value_optimizer.step()
        return value_loss.item()

    def update_policy(self, states, actions, returns, old_log_prob):
        losses = []
        advantages = self.estimate_advantage(returns, states)
        for i in range(self.policy_epochs):
            losses.append(self.policy_step(states, actions, old_log_prob, advantages))
        self.history.update_policy_losses(losses)

    def update_value_model(self, states, returns):
        losses = []
        for i in range(self.value_epochs):
            losses.append(self.value_step(states, returns))
        self.history.update_value_losses(losses)

    def update(self):
        states, actions, returns, logprobs = self.envs.get_train_data()

        states = Variable(states)
        actions = Variable(actions)
        returns = Variable(returns)
        logprobs = Variable(logprobs.data)

        self.update_policy(states, actions, returns, logprobs)
        self.update_value_model(states, returns)
        return str(self.history), len(states)

    def save(self, path):
        out = {'attrs': self.params}
        state_dicts = {}

        state_dicts['policy_model'] = self.policy_model.state_dict()
        state_dicts['value_model'] = self.value_model.state_dict()
        state_dicts['policy_optimizer'] = self.policy_optimizer.state_dict()
        state_dicts['value_optimizer'] = self.value_optimizer.state_dict()

        out['state_dicts'] = state_dicts
        torch.save(out, path)

    def run_test_episode(self, env, render=True):
        reward = 0
        state = env.reset()
        done = False
        while not done:

            with torch.no_grad():
                state = torch.FloatTensor(state).view(1, -1)
                action = self.policy_model(state)

            state, r, done, _ = env.step(np.asarray(action)[0])
            reward += r
            env.render()

        return reward


class PPOCombiAgent(PPOAgent):

    @classmethod
    def from_saved(cls, path, env_manager, model, optimizer):
        cp = torch.load(path)
        agent = cls(env_manager, model, optimizer)
        for k, v in cp['attrs'].items():
            setattr(agent, k, v)

        for k, v in cp['state_dicts'].items():
            getattr(agent, k).load_state_dict(v)
        return agent

    def __init__(self, env_manager, model, optimizer, epochs=20, grad_norm=0.5, normalize_advantage=True, clip_epsilon=.2, value_loss_coef=.1):
        self.value_loss_coef = value_loss_coef
        self.envs = env_manager
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.grad_norm = grad_norm
        self.normalize_advantage = normalize_advantage
        self.clip_epsilon = clip_epsilon
        self.episode_rewards = []
        self.episode = 0

    @property
    def params(self):
        return {k: v for k, v in self.__dict__.items() if not hasattr(v, '__dict__')}

    def episode_logstr(self, n, loss, policy_loss, value_loss):
        eps_logstr = 'episode {}, reward avg {:.2f} | samples: {} --  loss: {:.4f} | policy loss: {:.4f} |value loss: {:.4f}'
        return eps_logstr.format(self.episode, self.episode_rewards[-1],
                                 n, loss, policy_loss, value_loss)

    def run_episode(self, render=False):
        self.envs.reset()
        while not self.envs.ready:
            with torch.no_grad():
                actions, logprobs, values = self.model.sample_action(self.envs.current_states)
            self.envs.step(actions, logprobs, values)

        self.episode_rewards.append(self.envs.mean_reward)
        n, loss, policy_loss, value_loss = self.update()
        print(self.episode_logstr(n, loss, policy_loss, value_loss))
        self.episode += 1

    def estimate_advantage(self, returns, values):
        advantages = Variable(returns).view(-1, 1) - Variable(values.data)
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        return advantages

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

        advantages = self.estimate_advantage(returns, pred_values)

        for i in range(self.epochs):
            log_probs, values = self.model.evaluate_actions(states, actions)
            policy_loss = self.policy_loss(log_probs, old_logprobs, advantages)
            value_loss = self.value_loss(returns, values)

            self.optimizer.zero_grad()
            loss = (value_loss * self.value_loss_coef + policy_loss)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.grad_norm)
            self.optimizer.step()
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            losses.append(loss.item())
        return len(states), np.mean(losses), np.mean(policy_losses), np.mean(value_losses)

    def save(self, path):
        out = {'attrs': self.params}
        state_dicts = {}

        state_dicts['model'] = self.model.state_dict()
        state_dicts['optimizer'] = self.policy_optimizer.state_dict()
        out['state_dicts'] = state_dicts
        torch.save(out, path)

    def run_test_episode(self, env, render=True):
        reward = 0
        state = env.reset()
        done = False
        while not done:

            with torch.no_grad():
                state = torch.FloatTensor(state).view(1, -1)
                action, _ = self.model(state)

            state, r, done, _ = env.step(np.asarray(action)[0])
            reward += r
            env.render()

        return reward
