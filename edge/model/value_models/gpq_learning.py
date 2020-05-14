import numpy as np

from .. import GPModel
from ..inference import MaternGP


class GPQLearning(GPModel):
    def __init__(self, env, step_size, discount_rate,
                 x_seed=None, y_seed=None, y_dim=None, gp_params=None):
        if gp_params is None:
            gp_params = {}
        if x_seed is None:
            x_seed = np.empty((0, env.stateaction_space.data_length))
        if y_seed is None and y_dim is None:
            raise ValueError('seed_y and y_seed cannot be both None')
        elif y_seed is None:
            if y_dim == 1:
                y_seed = np.empty((0,))
            else:
                y_seed = np.empty((0, y_dim))

        gp = MaternGP(x_seed, y_seed, **gp_params)
        super(GPQLearning, self).__init__(env, gp)
        self.step_size = step_size
        self.discount_rate = discount_rate

    def update(self, state, action, new_state, reward):
        q_value_step = self.step_size * (
            reward + self.discount_rate * np.max(self[new_state, :])
        )
        q_value_update = self[state, action] + q_value_step

        stateaction = self.env.stateaction_space[state, action]
        self.gp = self.gp.append_data(stateaction, q_value_update)
