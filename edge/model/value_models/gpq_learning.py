import numpy as np
from pathlib import Path
import json

from .. import GPModel
from ..inference import MaternGP


class GPQLearning(GPModel):
    """
    Models the Q-Function over a StateActionSpace as a MaternGP updated with a Q-Learning update.
    """
    def __init__(self, env, step_size, discount_rate,
                 x_seed, y_seed, gp_params=None):
        """
        Initializer
        :param env: the environment
        :param step_size: the step size in the Q-Learning update
        :param discount_rate: the discount rate
        :param x_seed: the seed input of the GP. Typically a list of stateactions
        :param y_seed: the seed output of the GP. Typically a list of floats
        :param gp_params: the parameters defining the GP. See edge.models.inference.MaternGP for more information
        """
        if gp_params is None:
            gp_params = {}

        gp = MaternGP(x_seed, y_seed, **gp_params)
        super(GPQLearning, self).__init__(env, gp)
        self.step_size = step_size
        self.discount_rate = discount_rate

    def update(self, state, action, new_state, reward, failed):
        """
        Updates the underlying GP with the Q-Learning update
        :param state: the previous state
        :param action: the action taken
        :param new_state: the new state
        :param reward: the reward incurred
        :param failed: whether the agent has failed
        """
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
        """
        Dictionary of the configuration of the model. Useful for saving purposed
        :return: dict: the model's configuration
        """
        return {
            'step_size': self.step_size,
            'discount_rate': self.discount_rate
        }

    @staticmethod
    def load(load_folder, env, x_seed, y_seed):
        """
        Loads the model and the GP saved by the GPModel.save method. Note that this method may fail if the save was
        made with an older version of the code.
        :param load_folder: the folder where the files are
        :return: GPQLearning: the model
        """
        load_path = Path(load_folder)
        gp_load_path = str(load_path / GPModel.GP_SAVE_NAME)
        model_load_path = str(load_path / GPModel.SAVE_NAME)

        gp = MaternGP.load(gp_load_path)
        with open(model_load_path, 'r') as f:
            state_dict = json.load(f)

        model = GPQLearning(env, x_seed=x_seed, y_seed=y_seed, **state_dict)
        model.gp = gp

        return model
