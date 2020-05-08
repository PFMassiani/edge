import numpy as np

from .. import Model
from edge import error


class QLearning(Model):
    def __init__(self, stateaction_space, step_size, discount_rate):
        super(QLearning, self).__init__(stateaction_space)
        self.step_size = step_size
        self.discount_rate = discount_rate
        self.q_values = np.zeros(self.space.discretization_shape,
                                 dtype=np.float)

    def update(self, state, action, new_state, reward):
        stateaction = self.space.get_stateaction(state, action)
        if not self.space.is_on_grid(stateaction):
            raise error.NotOnGrid

        index = self.space.get_index_of(stateaction)
        new_index = self.space.state_space.get_index_of(new_state)

        Q[index] =
