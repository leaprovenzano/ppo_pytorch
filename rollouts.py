import torch
import numpy as np
import gym


class EnvWorker(object):

    """Episode based, acts as an interface with enviornment and
    tracks episode state, actions, etc for training on completion.

    episode memory objects (states, actions, rewards, logprobs) will
    be kept for episode duration and updated as lists. On episode completion
    they will be turned into torch tensors for training and the rewards
    will be modified using the self.reward_processor ( these operations are
    handled by the finalize method ). After training is complete we can 
    call the reset method to clean out memory  and start a new episode in 
    the env.


    Attributes:

        name (str): env name, this will be used to make our gym enviornment

        closed (bool): if true memory objects are closed and have been turned
            into torch tensors and rewards will have been processed. This worker
            is ready for training and will not take further steps until reset.

        current_state (torch.FloatTensor): tensor of the current state

        current_step (int): the current step in the enviorment if the worker is
            done this should also reflect the length of the memory objects

        done (bool): If true the episode has been completed in the enviorment no
            further enviornment steps will be taken until we call reset.

        env (gym enviornment): actual gym enviorment will be created at init given
            name using gym.make.

        episode_reward (int|float):  running reward for the current episode
            on episode completion (and call to finalize) this value will be stored
            in self.total_reward_log. On call to reset it is zeroed

        reward_processor (ppo_pytorch.utils.RewardProcessor): RewardProcessor object 
            used to process rewards on episode completion.

        total_reward_log (list): list of unprocessed final rewards for all the workers
            episodes. updated with each new episode from the episode_reward attribute on
            finalize

    Memory Attributes:

        actions (list | torch.tensor): actions taken during current
            episode. deleted on reset

        logprobs (list | torch.tensor): logprobs recorded during rollout from policy model 
            for actions taken in the current episode, deleted on reset.

        rewards (list   torch.tensor): during training this will be a list of tensors
            with unmodified rewards for each step directly from the enviornment. 
            After episode completion (and call to finalize) this attribute will be modified by 
            the reward processor to contain our processed reward tensor.

        states (list | torch.tensor): states recorded during training updated, deleted on reset.



    """

    def __init__(self, name, reward_processor):
        self.reward_processor = reward_processor
        self.total_reward_log = []

        self.name = name
        self.env = gym.make(name)

        self.reset()

    def __len__(self):
        return self.current_step

    def reset(self, *args):
        self.current_state = torch.FloatTensor(self.env.reset())
        self.current_step = 0
        self.episode_reward = 0
        self.done = False
        self.closed = False
        self.states = []
        self.actions = []
        self.rewards = []
        self.returns = []
        self.logprobs = []

    def get_current_state(self):
        return self.current_state

    def step(self, action, logprob):
        if not self.done:
            self.actions.append(action)
            self.logprobs.append(logprob)
            self.states.append(self.current_state)
            next_state, r, self.done, _ = self.env.step(np.asarray(action))
            self.current_state = torch.FloatTensor(next_state)
            self.episode_reward += r
            self.rewards.append(torch.FloatTensor([r]))
            self.current_step += 1
        self.finalize()

    def finalize(self):
        if self.done and (not self.closed):
            self.rewards = torch.cat(self.rewards)
            self.set_returns()
            self.states = torch.stack(self.states)
            self.actions = torch.stack(self.actions)
            self.logprobs = torch.stack(self.logprobs)
            self.total_reward_log.append(self.episode_reward)
            self.closed = True

    def set_returns(self):
        processed_rewards = self.reward_processor.shape(self.rewards)
        self.rewards = self.reward_processor.compute_discount_returns(
            processed_rewards)


class MultiEnvManager(object):

    """Rollout manager for multiple envworkers

    Attributes:
        envs (list (EnvWorker)): list of n envworkers
        n (int): number of envworkers to maintain
        name (str): env name, this will be used to instantiate our gym enviornments
        reward_processor (ppo_pytorch.utils.RewardProcessor): RewardProcessor object 
            passed to envworkers and used to process rewards on episode completion
    """

    def __init__(self, name, n, reward_processor):
        self.name = name
        self.n = n
        self.reward_processor = reward_processor

        self.envs = [EnvWorker(self.name, self.reward_processor)
                     for i in range(self.n)]

    def __len__(self):
        return self.n

    def __index__(self, i):
        return self.envs[i]

    def __getitem__(self, i):
        return self.envs[i]

    def reset(self):
        for env in self.envs:
            env.reset()

    @property
    def done(self):
        return [env.done for env in self.envs]

    @property
    def closed(self):
        return [env.closed for env in self.envs]

    @property
    def allclosed(self):
        return all(self.closed)

    def get_open(self):
        return [env for env in self.envs if not env.closed]

    @property
    def ready(self):
        return self.allclosed

    @property
    def current_states(self):
        return torch.stack([env.current_state for env in self.get_open()])

    @property
    def mean_reward(self):
        return np.mean([env.episode_reward for env in self.envs])

    def step(self, actions, logprobs):
        for i, env in enumerate(self.get_open()):
            env.step(actions[i], logprobs[i])

    def get_train_data(self):

        if self.ready:
            states = []
            actions = []
            rewards = []
            logprobs = []

            for env in self.envs:
                states.append(env.states)
                actions.append(env.actions)
                rewards.append(env.rewards)
                logprobs.append(env.logprobs)

            states = torch.cat(states)
            actions = torch.cat(actions)
            rewards = torch.cat(rewards)
            logprobs = torch.cat(logprobs)
            return states, actions, rewards, logprobs


