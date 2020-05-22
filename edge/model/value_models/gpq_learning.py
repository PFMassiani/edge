import numpy as np
from pathlib import Path
import json

from .. import GPModel
from ..inference import MaternGP


class GPQLearning(GPModel):
    def __init__(self, env, step_size, discount_rate,
                 x_seed, y_seed, gp_params=None):
        if gp_params is None:
            gp_params = {}

        gp = MaternGP(x_seed, y_seed, **gp_params)
        super(GPQLearning, self).__init__(env, gp)
        self.step_size = step_size
        self.discount_rate = discount_rate

    def update(self, state, action, new_state, reward, failed):
        current_value = self[state, action]
        q_value_step = self.step_size * (
            reward + self.discount_rate * np.max(self[new_state, :])
            - current_value
        )
        q_value_update = current_value + q_value_step

        stateaction = self.env.stateaction_space[state, action]
        self.gp = self.gp.append_data(stateaction, q_value_update)

    @property
    def state_dict(self):
        return {
            'step_size': self.step_size,
            'discount_rate': self.discount_rate
        }

    @staticmethod
    def load(load_folder, env, x_seed, y_seed):
        load_folder = Path(load_path)
        gp_load_path = str(load_path / GPModel.GP_SAVE_NAME)
        model_load_path = str(load_path / GPModel.SAVE_NAME)

        gp = MaternGP.load(gp_load_path)
        with open(model_load_path, 'r') as f:
            state_dict = json.load(f)

        model = GPQLearning(env, x_seed=x_seed, y_seed=y_seed, **state_dict)
        model.gp = gp

        return model
