import gpytorch

from edge.utils import atleast_2d, constraint_from_tuple
from .inference import GP
from .tensorwrap import tensorwrap


class MaternGP(GP):
    @tensorwrap('train_x', 'train_y')
    def __init__(self, train_x, train_y, nu=2.5,
                 noise_prior=None, noise_constraint=(1e-3, 1e4),
                 lengthscale_prior=None, lengthscale_constraint=None,
                 outputscale_prior=None, outputscale_constraint=None,
                 hyperparameters_initialization=None):
        train_x = atleast_2d(train_x)

        self.__structure_dict = {
            'nu': nu,
            'noise_prior': noise_prior,
            'noise_constraint': noise_constraint,
            'lengthscale_prior': lengthscale_prior,
            'lengthscale_constraint': lengthscale_constraint,
            'outputscale_prior': outputscale_prior,
            'outputscale_constraint': outputscale_constraint
        }

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
                                       covar_module, likelihood)

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

        self.initialize(**initialization)

    @property
    def structure_dict(self):
        return self.__structure_dict
