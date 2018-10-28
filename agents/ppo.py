from ppo_pytorch.agents.agent import Agent
import numpy as np

import torch
from torch import nn

from ppo_pytorch.utils import TrainHistory
from torch.autograd import Variable


class PPOAgent(Agent):

    @classmethod
    def from_saved(cls, path, env, memory, policy_model, value_model,  policy_optimizer, value_optimizer):
        cp = torch.load(path)
        agent = cls(env, memory, policy_model, value_model,
                    policy_optimizer, value_optimizer)
        for k, v in cp['attrs'].items():
            setattr(agent, k, v)

        for k, v in cp['state_dicts'].items():
            getattr(agent, k).load_state_dict(v)
        return agent




    def __init__(self, env, memory, policy_model, value_model,  policy_optimizer, value_optimizer, policy_epochs=20, value_epochs=20,
                 policy_grad_norm=0.5, value_grad_norm=0.5, normalize_advantage=True, clip_epsilon=.2):

        super(PPOAgent, self).__init__(env, memory)

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
        self.history = TrainHistory()


    def step(self, state, render=True):
        action, logprob = self.get_action(state)
        next_state, reward, done, _ = self.env.step(np.asarray(action[0]))
        if render:
            self.env.render()
        self.memory.update(state, action, reward, logprob)
        return next_state, done


    def get_action(self, state):
        with torch.no_grad():
            action, logprob = self.policy_model.sample_action(torch.FloatTensor(state).view(1, -1))
        return action, logprob

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
        # current log prob
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
        # self.value_model.eval()
        self.history.update_value_losses(losses)

    # def update(self):
    #     states, actions, returns, logprobs = self.memory.get_values()
    #     states = Variable(torch.cat(states))
    #     actions = Variable(torch.cat(actions))
    #     returns = Variable(returns)
    #     logprobs = Variable(logprobs.data)
    #     self.update_policy(states, actions, returns, logprobs)
    #     self.update_value_model(states, returns)
    #     return str(self.history)


    def update(self):
        states, actions, returns, logprobs = self.memory.get_values()
        # states , actions, returns memory.get_values()
        states = Variable(torch.cat(states))
        actions = Variable(torch.cat(actions))
        returns = Variable(returns)
        logprobs = Variable(logprobs.data)
        self.update_policy(states, actions, returns, logprobs)
        self.update_value_model(states, returns)
        return str(self.history)

    def save(self, path):
        out = {'attrs':self.params}
        state_dicts = {}

        state_dicts['policy_model'] = self.policy_model.state_dict()
        state_dicts['value_model'] = self.value_model.state_dict()
        state_dicts['policy_optimizer'] = self.policy_optimizer.state_dict()
        state_dicts['value_optimizer'] = self.value_optimizer.state_dict()

        out['state_dicts'] = state_dicts
        torch.save(out, path)
