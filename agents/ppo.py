from ppo_pytorch.agents.agent import Agent


class PPOAgent(Agent):

    def step(self, state, render=True):
        action, logprob = self.policy.get_action(state)
        next_state, reward, done, _ = self.env.step(np.asarray(action[0]))
        if render:
            self.env.render()
        self.memory.update(state, action, reward, logprob)
        return next_state, done
