from pathlib import Path
import numpy as np

from .. import GPModel
from ..inference import MaternGP


class GPSARSA(GPModel):
    """
    Models the Q-function as a GP whose dataset is composed of the rewards
    and whose posterior distribution is on the Q-function.
    See "Reinforcement learning with Gaussian processes", Engel et al., 2005 for
    details
    """
    def __init__(self, env, gp):
        super(GPSARSA, self).__init__(env, gp)

    def update(self, state, action, new_state, reward, failed, done):
        stateaction = self.env.stateaction_space[state, action]
        self.gp.append_data(stateaction, np.atleast_1d(reward),
                            is_terminal=np.atleast_1d(done))


class MaternGPSARSA(GPSARSA):
    def __init__(self, env, **gp_params):
        if gp_params.get('value_structure_discount_factor') is None:
            raise ValueError('The parameter `value_structure_discount_factor` '
                             'is mandatory for GPSARSA.')
        gp = MaternGP(**gp_params)
        super(MaternGPSARSA, self).__init__(env, gp)


    @property
    def state_dict(self):
        """
        Dictionary of the configuration of the model. Useful for saving purposed
        :return: dict: the model's configuration
        """
        return {}

    @staticmethod
    def load(load_folder, env, x_seed, y_seed, load_data=False):
        load_path = Path(load_folder)
        gp_load_path = str(load_path / GPModel.GP_SAVE_NAME)

        gp = MaternGP.load(gp_load_path, x_seed, y_seed, load_data)
        model = MaternGPSARSA(
            env=env,
            # Only basic arguments for GP construction: it is overwritten later
            x_seed=gp.train_x.cpu().numpy() if load_data else x_seed,
            y_seed=gp.train_y.cpu().numpy() if load_data else y_seed,
            value_structure_discount_factor=0.
        )
        model.gp = gp

        return model