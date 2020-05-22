import numpy as np

from .. import DiscreteModel


class QLearning(DiscreteModel):
    def __init__(self, env, step_size, discount_rate):
        super(QLearning, self).__init__(env)
        self.step_size = step_size
        self.discount_rate = discount_rate
        self.q_values = np.zeros(
            self.env.stateaction_space.index_shape,
            dtype=np.float
        )

    def update(self, state, action, new_state, reward):
        sa_index = (
            self.env.stateaction_space.state_space.get_index_of(state),
            self.env.stateaction_space.action_space.get_index_of(action)
        )

        self.q_values[sa_index] = self[sa_index] + self.step_size * (
            reward + self.discount_rate * np.max(self[new_state, :])
            - self.q_values[sa_index]
        )

    def query(self, index):
        return self.q_values[index]
