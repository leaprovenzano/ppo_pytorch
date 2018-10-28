import numpy as np
import gym


class Agent(object):

    def __init__(self, env, memory):
        if type(env) is str:
            self.set_env_from_envname(env)
        else:
            self.env = env
        self.memory = memory

        self.episode = 0
        self.episode_rewards = []


    @property
    def params(self):
        return {k: v for k, v in self.__dict__.items() if not hasattr(v, '__dict__')}

    def get_action(*args, **kwargs):
        raise NotImplementedError()

    def set_env_from_envname(self, env_name):
        self.env = gym.make(env_name)

    def train(self):
        if self.memory.is_open:
            self.memory.finalize()
        train_logstr = self.update()
        return train_logstr

    def episode_logstr(self, train_stats):
        eps_logstr = 'episode {}, reward {:.2f} reward_avg {:.2f} | train stats : {}'
        return eps_logstr.format(self.episode, self.episode_rewards[-1],
                                 np.mean(self.episode_rewards[-100:]),
                                 train_stats)

    def step(self, state, render=True):
        action = self.get_action(state)
        next_state, reward, done, _ = self.env.step(np.asarray(action[0]))
        if render:
            self.env.render()
        self.memory.update(state, action, reward)
        return next_state, done

    def run_episode(self, render=True):
        self.memory.reset()

        state = self.env.reset()

        done = False
        while not done:
            state, done = self.step(state, render=render)

        self.episode_rewards.append(self.memory.total_reward)
        update_stats = self.train()

        print(self.episode_logstr(update_stats))
        self.episode += 1

    # def save(self, path):
    #     cp = {'episode': self.episode,
    #                    'episode_rewards' : self.episode_rewards,
    #                    'env' : self.env.spec.id,
    #                    'policy' : self.policy.serialize()
    #                    }

    #          #  'baseline_grad_norm' : agent.baseline_grad_norm,
    #          #   'gamma' : agent.gamma,
    #          #   'policy' :agent.policy.state_dict(),
    #          #   'baseline': agent.baseline.state_dict(),
    #          #   'policy_optimizer' : agent.policy_optimizer.state_dict(),
    #          #   'baseline_optimizer' : agent.baseline_optimizer.state_dict()
    #          # }

    #     torch.save(cp, path)
