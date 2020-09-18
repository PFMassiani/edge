import gpytorch
import torch

from edge.utils import atleast_2d, constraint_from_tuple
from .inference import GP
from .tensorwrap import tensorwrap, ensure_tensor


class MaternGP(GP):
    """
    Specializes the GP class with a Matern covariance kernel, a Zero mean, and some priors on the hyperparameters
    """
    @tensorwrap('train_x', 'train_y')
    def __init__(self, train_x, train_y, nu=2.5,
                 noise_prior=None, noise_constraint=(1e-3, 1e4),
                 lengthscale_prior=None, lengthscale_constraint=None,
                 outputscale_prior=None, outputscale_constraint=None,
                 hyperparameters_initialization=None,
                 dataset_type=None, dataset_params=None):
        """
        Initializer
        :param train_x: training input data. Should be 2D, and interpreted as a list of points.
        :param train_y: np.ndarray: training output data. Should be 1D, or of shape (train_x.shape[0], 1).
        :param nu: the nu parameter of the Matern kernel. Depends on the desired smoothness
        :param noise_prior: None or (mu, sigma). If not None, the noise has a prior NormalPrior(mu, sigma)
        :param noise_constraint: None or (nMin, nMax). If not None, the noise is bounded between nMin and nMax
        :param lengthscale_prior: None or (mu, sigma). If not None, the lengthscale has a prior NormalPrior(mu, sigma)
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
        train_x = atleast_2d(train_x)

        self.__structure_dict = {
            'nu': nu,
            'noise_prior': noise_prior,
            'noise_constraint': noise_constraint,
            'lengthscale_prior': lengthscale_prior,
            'lengthscale_constraint': lengthscale_constraint,
            'outputscale_prior': outputscale_prior,
            'outputscale_constraint': outputscale_constraint,
            'dataset_type': dataset_type,
            'dataset_params': dataset_params,
        }

        # Using a ConstantMean here performs much worse than a ZeroMean
        mean_module = gpytorch.means.ZeroMean()

        if lengthscale_prior is not None:
            lengthscale_prior = gpytorch.priors.NormalPrior(*lengthscale_prior)
        lengthscale_constraint = constraint_from_tuple(lengthscale_constraint)

        if outputscale_prior is not None:
            outputscale_prior = gpytorch.priors.NormalPrior(*outputscale_prior)
        outputscale_constraint = constraint_from_tuple(outputscale_constraint)

        ard_num_dims = train_x.shape[1]

        covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=nu,
                ard_num_dims=ard_num_dims,
                lengthscale_prior=lengthscale_prior,
                lengthscale_constraint=lengthscale_constraint
            ),
            outputscale_prior=outputscale_prior,
            outputscale_constraint=outputscale_constraint
        )

        if noise_prior is not None:
            noise_prior = gpytorch.priors.NormalPrior(*noise_prior)
        noise_constraint = constraint_from_tuple(noise_constraint)

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=noise_prior,
            noise_constraint=noise_constraint
        )

        super(MaternGP, self).__init__(train_x, train_y, mean_module,
                                       covar_module, likelihood, dataset_type, dataset_params)

        initialization = {}
        if noise_prior is not None:
            initialization['likelihood.noise_covar.noise'] = noise_prior.mean
        if outputscale_prior is not None:
            initialization['covar_module.outputscale'] = outputscale_prior.mean
        if lengthscale_prior is not None:
            initialization['covar_module.base_kernel.lengthscale'] = \
                lengthscale_prior.mean

        if hyperparameters_initialization is not None:
            initialization.update(hyperparameters_initialization)

        initialization = {k: ensure_tensor(v) for k, v in initialization.items()}

        self.initialize(**initialization)

    @property
    def structure_dict(self):
        return self.__structure_dict
