import numpy as np

from .. import Model


class QLearning(Model):
    def __init__(self, stateaction_space, step_size, discount_rate):
        super(QLearning, self).__init__(stateaction_space)
        self.step_size = step_size
        self.discount_rate = discount_rate
        self.q_values = np.zeros(self.space.index_shape, dtype=np.float)

    def update(self, state_index, action_index, new_state_index, reward):
        sa_index = (state_index, action_index)

        self.q_values[sa_index] = self.q_values[sa_index] + self.step_size * (
            reward + self.discount_rate * np.max(
                self.q_values[new_state_index, :]
            )
        )

    def query(self, state, action):
        stateaction = self.space.get_stateaction(state, action)
        index = self.space.get_index_of(stateaction)
        return self.q_values[index]
