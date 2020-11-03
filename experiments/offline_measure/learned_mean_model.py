import logging
from pathlib import Path
from gpytorch.means import Mean

from edge.utils import atleast_2d, get_hyperparameters
from edge.model.inference import MaternGP, GP
from edge.model.inference.tensorwrap import tensorwrap

from edge.model.safety_models import SafetyMeasure


def mean_from_gp(gp, param_name):
    try:
        params = get_hyperparameters(gp, constraints=False, around=None)
        stripped_keys_params = {
            key.split('.')[-1]: value for key, value in params.items()
        }
        return stripped_keys_params[param_name]
    except KeyError:
        logging.error(f'Could not load attribute {param_name} from {gp}')
        return None


class GPMean(Mean):
    def __init__(self, gp):
        super().__init__()
        self.gp = gp

    def forward(self, x):
        return self.gp.predict(x).mean


def final_param_value(mean_module, param_name, param):
    if param is None:
        return (mean_from_gp(mean_module, param_name), 1.)
    else:
        return param


def system_agnostic_mean_path(mean_path):
    ROOT_DIR = Path(__file__).parent.parent.parent.parent
    mean_path = str(mean_path)
    mean_path = mean_path.split('edge')[-1]
    return ROOT_DIR / mean_path


class GPMeanMaternGP(MaternGP):
    @tensorwrap('train_x', 'train_y')
    def __init__(self, train_x, train_y, mean_path, nu=2.5,
                 noise_prior=None, noise_constraint=(1e-3, 1e4),
                 lengthscale_prior=None, lengthscale_constraint=None,
                 outputscale_prior=None, outputscale_constraint=None,
                 hyperparameters_initialization=None,
                 dataset_type=None, dataset_params=None,
                 value_structure_discount_factor=None, **kwargs):
        """
        Initializer
        :param train_x: training input data. Should be 2D, and interpreted as a list of points.
        :param train_y: np.ndarray: training output data. Should be 1D, or of shape (train_x.shape[0], 1).
        :param mean_path: Path: path to the GP loaded as a mean_module. It should
            be saved with its data
        :param nu: the nu parameter of the Matern kernel. Depends on the desired smoothness
        :param noise_prior: None or (mu, sigma). If not None, the noise has a prior NormalPrior(mu, sigma)
        :param noise_constraint: None or (nMin, nMax). If not None, the noise is bounded between nMin and nMax
        :param lengthscale_prior: None or (mu, sigma) or ((mu_i, sigma_i)):
            * if (mu, sigma), the lengthscale has a prior NormalPrior(mu, sigma)
            * if ((mu_i, sigma_i)), the lengthscale has a
                MultivariateNormalPrior, with mean diag(mu_i) and covariance
                diag(sigma_i)
        :param lengthscale_constraint: None or (lMin, lMax). If not None, the lengthscale is bounded between lMin and
            lMax
        :param outputscale_prior: None or (mu, sigma). If not None, the outputscale has a prior NormalPrior(mu, sigma)
        :param outputscale_constraint: None or (oMin, oMax). If not None, the outputscale is bounded between oMin and
            oMax
        :param hyperparameters_initialization: None or dict. The hyperparameters are initialized to the values
            specified. The remaining ones are either initialized as their prior mean, or left uninitialized if no prior
            is specified.
        :param dataset_type: If 'timeforgetting', use a TimeForgettingDataset. Otherwise, a default Dataset is used
        :param dataset_params: dictionary or None. The entries are passed as keyword arguments to the constructor of
            the chosen dataset.
        """
        self.__structure_dict = {
            'mean_path': mean_path
        }

        train_x = atleast_2d(train_x)
        mean_gp = GP.load(system_agnostic_mean_path(mean_path),
                          train_x, train_y, load_data=True)
        lengthscale_prior = final_param_value(
            mean_gp, 'lengthscale', lengthscale_prior
        )
        outputscale_prior = final_param_value(
            mean_gp, 'outputscale', outputscale_prior
        )
        noise_prior = final_param_value(
            mean_gp, 'noise', noise_prior
        )

        super_dict = {
            'train_x': train_x,
            'train_y': train_y,
            'nu': nu,
            'noise_prior': noise_prior,
            'noise_constraint': noise_constraint,
            'lengthscale_prior': lengthscale_prior,
            'lengthscale_constraint': lengthscale_constraint,
            'outputscale_prior': outputscale_prior,
            'outputscale_constraint': outputscale_constraint,
            'hyperparameters_initialization': hyperparameters_initialization,
            'mean_constant': None,
            'dataset_type': dataset_type,
            'dataset_params': dataset_params,
            'value_structure_discount_factor': value_structure_discount_factor,
        }

        super().__init__(**super_dict)
        self.mean_module = GPMean(mean_gp)

    @property
    def structure_dict(self):
        structure_dict = super().structure_dict.copy()
        structure_dict.pop('mean_constant')
        structure_dict.update(self.__structure_dict)
        return structure_dict

    @staticmethod
    def load(load_path, train_x, train_y, load_data=False):
        return MaternGP.load(
            load_path, train_x, train_y, load_data, GPMeanMaternGP
        )

class LearnedMeanMaternSafety(SafetyMeasure):
    def __init__(self, env, gamma_measure, x_seed, y_seed, gp_params=None):
        if gp_params is None:
            gp_params = {}
        gp = GPMeanMaternGP(x_seed, y_seed, **gp_params)
        super().__init__(env, gp, gamma_measure)

    @staticmethod
    def load(load_folder, env, gamma_measure, x_seed, y_seed):
        load_path = Path(load_folder)
        gp_load_path = str(load_path / LearnedMeanMaternSafety.GP_SAVE_NAME)
        gp = GPMeanMaternGP.load(gp_load_path, x_seed, y_seed)
        model = LearnedMeanMaternSafety(env, gamma_measure=gamma_measure,
                                        x_seed=x_seed, y_seed=y_seed,
                                        gp_params=gp.structure_dict)
        model.gp = gp
        return model
