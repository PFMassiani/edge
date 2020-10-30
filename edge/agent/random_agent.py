import numpy as np

from . import Agent
from edge.model.policy_models import RandomPolicy


class RandomAgent(Agent):
    def __init__(self, env):
        super(RandomAgent, self).__init__(env)
        self.policy = RandomPolicy(env.stateaction_space)

    def get_next_action(self):
        return self.policy.get_action(self.state)

    def update_models(self, state, action, new_state, reward, failed, done):
        pass

    def fit_models(self):
        pass