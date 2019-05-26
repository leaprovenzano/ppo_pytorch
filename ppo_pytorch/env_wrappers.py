import numpy as np
import torch

import gym
from gym import Wrapper
from ppo_pytorch.utils import MinMaxScaler


class ActionRange(object):

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if not instance.is_discrete:
            high = instance.action_space.high
            low = instance.action_space.low
            if self._all_finite(low, high):
                if self._all_reducable(low, high):
                    return float(low[0]), float(high[0])
                else:
                    return torch.FloatTensor(low), torch.FloatTensor(high)

        return None

    @staticmethod
    def _is_finite(vals):
        return all(np.isfinite(vals))

    def _all_finite(self, low, high):
        return self._is_finite(low) and self._is_finite(high)

    @staticmethod
    def _is_reducable(vals) -> bool:
        return all(vals == vals[0])

    def _all_reducable(self, low, high):
        return self._is_reducable(low) and self._is_reducable(high)

    @staticmethod
    def all_finite(self, low, high):
        return self.is_finite(low) and self.is_finite(high)


class TensorEnvWrapper(Wrapper):

    action_range = ActionRange()

    def __init__(self, env):
        super(TensorEnvWrapper, self).__init__(env)
        self.is_discrete = type(self.env.action_space.sample()) is int
        if self.is_discrete:
            self.env.action_space.shape = (self.env.action_space.n,)

        self.current_state = None
        self.total_reward = 0
        self.done = False
        self.reset()

    @property
    def info(self):
        return dict(name=self.env.spec.id,
                    wrapper=self.__class__.__name__,
                    action_range=self.action_range,
                    action_shape=self.env.action_space.shape,
                    observation_shape=self.env.observation_space.shape,
                    discrete=self.is_discrete,
                    done=self.done,
                    total_reward=self.total_reward,
                    current_state=self.current_state
                    )

    @staticmethod
    def tensorize(x):
        return torch.FloatTensor(x)

    def action(self, action):
        if self.is_discrete:
            return action.item()
        return np.asarray(action)

    def observation(self, observation):
        return self.tensorize(observation)

    def reward(self, reward):
        self.total_reward += reward
        reward = self.tensorize([reward])
        return reward

    def step(self, action, *args, **kwargs):
        action = self.action(action)
        observation, reward, done, info = self.env.step(action)
        self.current_state = self.observation(observation)
        self.done = done
        return self.current_state, self.reward(reward), torch.FloatTensor([done]), info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.current_state = self.observation(observation)
        self.total_reward = 0.
        self.done = False
        return self.current_state

    def sample(self):
        a = self.env.action_space.sample()
        if self.is_discrete:
            return torch.IntTensor([a])
        return torch.FloatTensor(a)


class ScaledObservationWrapper(TensorEnvWrapper):
    """ObservationWrapper for openai gym's atari  RAM envs. just scales the observation between
    an observation_range
    """

    def __init__(self, env, observation_range=(-1, 1)):
        self.scaler = MinMaxScaler((env.observation_space.low, env.observation_space.high), observation_range)
        super(ScaledObservationWrapper, self).__init__(env)

    def rescale(self, observation):
        return self.scaler(observation)

    def observation(self, observation):
        return self.tensorize(self.rescale(observation))


class YOLOAtariWrapper(ScaledObservationWrapper):
    """ObservationWrapper for openai gym's atari RAM envs. This wrapper returns done
    on the loss of an life in the atari env ( so here YOLO is you only live once)
    """

    def __init__(self, env, *args, **kwargs):
        super(YOLOAtariWrapper, self).__init__(env, *args, **kwargs)

    @property
    def ale_lives(self):
        return self.unwrapped.ale.lives()

    def step(self, action, *args, **kwargs):
        action = self.action(action)
        ale_lives = self.ale_lives
        observation, reward, done, info = self.env.step(action)
        self.current_state = self.observation(observation)
        done = (self.ale_lives < ale_lives) or done
        self.done = done
        return self.current_state, self.reward(reward), done, info


class MultiEnv(object):

    @staticmethod
    def make_build_func(name, wrapper):
        def build_func():
            return wrapper(gym.make(name))
        return build_func

    def __init__(self, name, n, wrapper=TensorEnvWrapper):
        self.name = name
        self.n = n
        self._factory = self.make_build_func(name, wrapper)
        self.envs = self._build()
        self.render_index = 0

    def __len__(self, i):
        return len(self.envs)

    def __getitem__(self, i):
        return self.envs[i]

    def __getattr__(self, k):
        return getattr(self.envs[0], k)

    def _build(self):
        return [self._factory() for i in range(self.n)]

    @property
    def current_state(self):
        return torch.stack([e.current_state for e in self.envs])

    @property
    def done(self):
        return torch.Tensor([e.done for e in self.envs]).reshape(-1, 1)

    @property
    def done_indices(self):
        return [i for i, e in enumerate(self.envs) if e.done]

    @property
    def total_reward(self):
        return torch.Tensor([e.total_reward for e in self.envs])

    def render(self):
        return self.envs[self.render_index].render()

    def sample(self):
        return torch.stack([env.sample() for env in self.envs])

    def step(self, actions):
        rewards = []
        infos = []
        for action, env in zip(actions, self.envs):
            _, r, _, inf = env.step(action)
            rewards.append(r)
            infos.append(inf)
        return self.current_state, torch.stack(rewards), self.done, infos

    def soft_reset(self):
        total_rewards = []
        for i in self.done_indices:
            total_rewards.append(self.envs[i].total_reward)
            self.envs[i].reset()
        return total_rewards

    def reset(self):
        for env in self.envs:
            env.reset()
        return self.current_state

    def close(self):
        for env in self.envs:
            env.close()
