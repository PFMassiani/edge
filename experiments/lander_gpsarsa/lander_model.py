from pathlib import Path

from edge.model.value_models.gpsarsa import GPSARSA
from edge.model.inference.custom_kernels_gps import SymmetricMaternCosGP
from edge.model.safety_models import SafetyMeasure


class SymmetricMaternCosGPSARSA(GPSARSA):
    def __init__(self, env, **gp_params):
        if gp_params.get('value_structure_discount_factor') is None:
            raise ValueError('The parameter `value_structure_discount_factor` '
                             'is mandatory for GPSARSA.')
        gp = SymmetricMaternCosGP(**gp_params)
        super(SymmetricMaternCosGPSARSA, self).__init__(env, gp)


    @property
    def state_dict(self):
        """
        Dictionary of the configuration of the model. Useful for saving purposed
        :return: dict: the model's configuration
        """
        return {}

    @staticmethod
    def load(load_folder, env, x_seed, y_seed):
        load_path = Path(load_folder)
        gp_load_path = str(load_path / SymmetricMaternCosGPSARSA.GP_SAVE_NAME)

        gp = SymmetricMaternCosGP.load(gp_load_path, x_seed, y_seed)
        model = SymmetricMaternCosGPSARSA(
            env=env,
            # Only basic arguments for GP construction: it is overwritten later
            train_x=x_seed,
            train_y=y_seed,
            value_structure_discount_factor=0.
        )
        model.gp = gp

        return model


class SymmetricMaternCosSafety(SafetyMeasure):
    def __init__(self, env, gamma_measure, x_seed, y_seed, gp_params=None):
        """
        Initializer
        :param env: the environment
        :param gamma_measure: the gamma coefficient used by the measure. It corresponds to gamma_optimistic
        :param x_seed: the seed input of the GP: a list of stateactions
        :param y_seed: the seed output of the GP: a list of floats
        :param gp_params: the parameters defining the GP. See edge.models.inference.MaternGP for more information
        """
        if gp_params is None:
            gp_params = {}
        gp = SymmetricMaternCosGP(x_seed, y_seed, **gp_params)
        super(SymmetricMaternCosSafety, self).__init__(env, gp, gamma_measure)

    @staticmethod
    def load(load_folder, env, gamma_measure, x_seed, y_seed):
        """
        Loads the model and the GP saved by the GPModel.save method. Note that this method may fail if the save was
        made with an older version of the code.
        :param load_folder: the folder where the files are
        :return: MaternSafety: the model
        """
        load_path = Path(load_folder)
        gp_load_path = str(load_path / SymmetricMaternCosSafety.GP_SAVE_NAME)

        gp = SymmetricMaternCosGP.load(gp_load_path, x_seed, y_seed)

        model = SymmetricMaternCosSafety(env, gamma_measure=gamma_measure,
                                         x_seed=x_seed, y_seed=y_seed)
        model.gp = gp

        return model