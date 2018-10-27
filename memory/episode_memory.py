import torch

class EpisodeMemory(object):

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.total_reward = 0
        self.is_open = True

    def __len__(self):
        return len(self.rewards)

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.total_reward = 0
        self.is_open = True

    def update(self, state, action, reward):
        self.states.append(torch.FloatTensor(state).view(1,-1))
        self.actions.append(action)
        self.rewards.append(torch.FloatTensor([reward]))
        self.total_reward += reward

    def finalize(self):
        self.states = torch.cat(self.states)
        self.actions = torch.cat(self.actions)
        self.rewards = torch.cat(self.rewards)
        self.is_open = False


class PPOEpisodeMemory(EpisodeMemory):

    def __init__(self, reward_processor):
        super(PPOEpisodeMemory, self).__init__()
        self.reward_processor = reward_processor
        self.logprobs = []

    def reset(self):
        super(PPOEpisodeMemory, self).reset()
        self.logprobs = []
        self.returns = None

    def update(self, state, action, reward, log_prob):
        super(PPOEpisodeMemory, self).update(state, action, reward)
        self.logprobs.append(log_prob)

    def finalize(self):
        super(PPOEpisodeMemory, self).finalize()
        self.logprobs = torch.cat(self.logprobs)
        self.set_returns()

    def set_returns(self):
        processed_rewards = self.reward_processor.shape(self.rewards)
        self.returns = self.reward_processor.compute_discount_returns(processed_rewards)

        

