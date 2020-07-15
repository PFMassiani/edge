import numpy as np

from . import Agent
from .policy import RandomPolicy


class RandomAgent(Agent):
    def __init__(self, env):
        super(RandomAgent, self).__init__(env)
        self.policy = RandomPolicy(env)

    def get_next_action(self):
        return self.policy.get_action(self.state)

    def update_models(self, state, action, new_state, reward, failed):
        pass

    def fit_models(self):
        pass