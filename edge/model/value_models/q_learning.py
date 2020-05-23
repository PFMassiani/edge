import numpy as np
from pathlib import Path
import json

from .. import DiscreteModel


class QLearning(DiscreteModel):
    ARRAY_SAVE_NAME = 'q_values_map.npy'
    SAVE_NAME = 'model.json'

    def __init__(self, env, step_size, discount_rate):
        super(QLearning, self).__init__(env)
        self.step_size = step_size
        self.discount_rate = discount_rate
        self.q_values = np.zeros(
            self.env.stateaction_space.index_shape,
            dtype=np.float
        )

    def update(self, state, action, new_state, reward, failed):
        sa_index = (
            self.env.stateaction_space.state_space.get_index_of(state),
            self.env.stateaction_space.action_space.get_index_of(action)
        )

        self.q_values[sa_index] = self[sa_index] + self.step_size * (
            reward + self.discount_rate * np.max(self[new_state, :])
            - self.q_values[sa_index]
        )

    def _query(self, index):
        return self.q_values[index]

    @property
    def state_dict(self):
        return {'step_size': self.step_size,
                'discount_rate': self.discount_rate}

    def save(self, save_folder):
        save_path = Path(save_folder)
        model_path = save_path / QLearning.SAVE_NAME
        array_path = save_path / QLearning.ARRAY_SAVE_NAME

        with model_path.open('w') as f:
            json.dump(self.state_dict, f, indent=4)

        np.save(array_path, self.q_values)

    @staticmethod
    def load(load_folder, env):
        load_path = Path(load_folder)
        model_path = load_path / QLearning.SAVE_NAME
        array_path = load_path / QLearning.ARRAY_SAVE_NAME

        with model_path.open('r') as f:
            state_dict = json.load(f)
        q_values = np.load(array_path)

        model = QLearning(env, **state_dict)
        model.q_values = q_values

        return model