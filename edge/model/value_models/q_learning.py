import numpy as np
from pathlib import Path
import json

from .. import DiscreteModel


class QLearning(DiscreteModel):
    """
    Models the Q-Function over a StateActionSpace as an array of values updated with Q-Learning. This is the vanilla
    Q-Learning.
    """
    ARRAY_SAVE_NAME = 'q_values_map.npy'
    SAVE_NAME = 'model.json'

    def __init__(self, env, step_size, discount_rate):
        """
        Initializer
        :param env: the environment
        :param step_size: the step size in the Q-Learning update
        :param discount_rate: the discount rate
        """
        super(QLearning, self).__init__(env)
        self.step_size = step_size
        self.discount_rate = discount_rate
        self.q_values = np.zeros(
            self.env.stateaction_space.index_shape,
            dtype=np.float
        )

    def update(self, state, action, new_state, reward, failed):
        """
        Updates the value of (state, action) with the Q-Learning update
        :param state: the previous state
        :param action: the action taken
        :param new_state: the new state
        :param reward: the reward incurred
        :param failed: whether the agent has failed
        """
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
        """
        Saves the model in the given folder. The array is saved in the file QLearning.ARRAY_SAVE_NAME, and the model
        itself in QLearning.SAVE_NAME
        :param save_folder: str or Path: the folder where to save
        """
        save_path = Path(save_folder)
        model_path = save_path / QLearning.SAVE_NAME
        array_path = save_path / QLearning.ARRAY_SAVE_NAME

        with model_path.open('w') as f:
            json.dump(self.state_dict, f, indent=4)

        np.save(array_path, self.q_values)

    @staticmethod
    def load(load_folder, env):
        """
        Loads the model and the array saved by the QLearning.save method. Note that this method may fail if the save was
        made with an older version of the code.
        :param load_folder: the folder where the files are
        :return: QLearning: the model
        """
        load_path = Path(load_folder)
        model_path = load_path / QLearning.SAVE_NAME
        array_path = load_path / QLearning.ARRAY_SAVE_NAME

        with model_path.open('r') as f:
            state_dict = json.load(f)
        q_values = np.load(array_path)

        model = QLearning(env, **state_dict)
        model.q_values = q_values

        return model