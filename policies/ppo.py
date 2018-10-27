import torch

from ppo_pytorch.utils import TrainHistory


class PPO(object):

    def __init__(self, policy_model, value_model,  policy_opitmizer, value_optimizer, policy_epochs=20,
                 policy_grad_norm=0.5, value_grad_norm=0.5, normalize_advantage=True, clip_epsilon=.2),

        self.policy_model = policy_model
        self.value_model = value_model
        self.policy_opitmizer = policy_opitmizer
        self.value_optimizer = value_optimizer
        self.policy_epochs = policy_epochs
        self.value_epochs = value_epochs
        self.policy_grad_norm = policy_grad_norm
        self.value_grad_norm = value_grad_norm
        self.normalize_advantage = normalize_advantage
        self.clip_epsilon = clip_epsilon
        self.history = TrainHistory()

    def get_action(self, state):
        with torch.no_grad():
            action = self.policy_model.sample_action(
                torch.FloatTensor(state).view(1, -1))
        return action

    def estimate_advantage(self, returns, states):
        advantages = returns.view(-1, 1) - self.value_model.predict(states).view(-1, 1)
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        return advantages

    def policy_loss(self, log_probs, old_log_probs, advantages):
        ratio = torch.exp(logprobs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        return -torch.min(surr1, surr2).mean()

    def value_loss(self, y_true, y_pred):
        return (0.5 * (y_true.view(-1, 1) - y_pred.view(-1, 1)).pow(2)).mean()

    def policy_step(self, states, actions, old_log_probs, advantages):
        # current log prob
        logprob = self.policy.evaluate_action(states, actions)
        policy_loss = self.policy_loss(log_probs, old_log_probs, advantages)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.policy_grad_norm)
        self.policy_optimizer.step()
        return policy_loss.item()

    def value_step(self, states, returns):
        value_pred = self.value_model.predict(states)
        value_loss = self.value_loss(returns, value_pred)
        self.baseline_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            self.baseline.parameters(), self.baseline_grad_norm)
        self.baseline_optimizer.step()
        return value_loss.item()

    def update_policy(self, states, actions, returns, old_log_probs):
        losses = []
        advantages = self.estimate_advantage(returns, states)
        self.policy_model.train()
        for i in range(self.policy_epochs):
            loss = self.policy_step(states, actions, old_log_probs, advantages)
            losses.append(loss)
        self.policy_model.eval()
        self.history.update_policy_losses(losses)

    def update_value_model(self, states, returns):
        losses = []
        self.value_model.train()
        for i in range(self.value_epochs):
            loss = self.value_step(states, returns)
            losses.append(loss)
        self.value_model.eval()
        self.history.update_value_losses(losses)

    def update(self, memory):
        self.update_policy(memory.states, memory.actions,
                           memory.returns, memory.log_probs)
        self.update_value_model(memory.states, memory.returns)
        return str(self.history)
