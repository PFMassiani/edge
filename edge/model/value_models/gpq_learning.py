import numpy as np

from .. import GPModel
from ..inference import MaternGP


class GPQLearning(GPModel):
    def __init__(self, stateaction_space, step_size, discount_rate,
                 seed_x, seed_y, gp_params=None):
        if gp_params is None:
            gp_params = {}
        gp = MaternGP(seed_x, seed_y, **gp_params)
        super(GPModel, self).__init__(stateaction_space, gp)

    def update(self, state, action, new_state, reward):
        q_value_step = self.step_size * (
            reward + self.discount_rate * np.max(self[new_state, :])
        )
        q_value_update = self[state, action] + q_value_step

        self.gp = self.gp.append_data(state, q_value_update)
