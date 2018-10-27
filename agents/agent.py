import numpy as np


class Agent(object):

    def __init__(self, env, policy, memory):

        self.env = env
        self.policy = policy
        self.memory = memory

        self.episode = 0
        self.episode_rewards = []

    def update(self):
        if self.memory.is_open:
            self.memory.finalize()
        train_logstr = self.policy.update(self.memory)
        return train_logstr

    def episode_logstr(self, train_stats):
        eps_logstr = 'episode {}, reward {:.2f} reward_avg {:.2f} | train stats : {}'
        return logstr.format(self.episode, self.episode_rewards[-1],
                             np.mean(self.episode_rewards[-100:]),
                             train_stats)

    def step(self, state, render=True):
        action = self.policy.get_action(state)
        next_state, reward, done, _ = self.env.step(np.asarray(action[0]))
        if render:
            self.env.render()
        self.memory.update(state, action, reward)
        return next_state, done

    def run_episode(self, render=True):

        state = self.env.reset()
        self.memory.reset()

        done = False
        while not done:
            state, done = self.step(state, render=render)

        self.episode_rewards.append(self.memory.total_reward)
        update_stats = self.update()

        print(self.episode_logstr)
        self.episode += 1
