import gpytorch
import math

from edge.utils import atleast_2d, constraint_from_tuple
from .inference import GP
from .custom_kernels import ConjugateKernel, ProductDecompositionKernel
from .tensorwrap import tensorwrap, ensure_tensor


def get_prior_and_constraint(prior, constraint):
    if prior is not None:
        prior = gpytorch.priors.NormalPrior(*prior)
    constraint = constraint_from_tuple(constraint)
    return prior, constraint

def get_initialization(name, initialization, prior):
    init_val = initialization.get(name)
    if init_val is not None:
        return init_val
    else:
        return prior.mean


class SymmetricMaternCosGP(GP):
    @tensorwrap('train_x', 'train_y')
    def __init__(self, train_x, train_y,
                 noise_prior=None, noise_constraint=(1e-3, 1e4),
                 lengthscale_prior=None, lengthscale_constraint=None,
                 outputscale_prior=None, outputscale_constraint=None,
                 hyperparameters_initialization=None,
                 **kwargs):
        train_x = atleast_2d(train_x)
        if train_x.shape[1] != 6:
            raise ValueError('SymmetricMaternCosGP can only be used on '
                             '6-dimensional data.')

        self.__structure_dict = {
            'noise_prior': noise_prior,
            'noise_constraint': noise_constraint,
            'lengthscale_prior': lengthscale_prior,
            'lengthscale_constraint': lengthscale_constraint,
            'outputscale_prior': outputscale_prior,
            'outputscale_constraint': outputscale_constraint,
        }
        self.__structure_dict.update(kwargs)

        initialization = {}
        ls_fact_I_pname = 'covar_module.base_kernel.{i}.kernels.0.lengthscale'
        ls_fact_I_conj_pname = 'covar_module.base_kernel.{i}.kernels.1.base_kernel.lengthscale'

        mean_module = gpytorch.means.ZeroMean()

        lp, lc = get_prior_and_constraint(lengthscale_prior,
                                          lengthscale_constraint)
        matern_0 = gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=4,
                lengthscale_prior=lp,
                lengthscale_constraint=lc,
        )
        l_init = get_initialization(
            'matern_0.lengthscale', hyperparameters_initialization, lp
        )
        matern_0.lengthscale = l_init
        sym_matern_0 = matern_0 + ConjugateKernel(
            matern_0, conjugation=[1, -1, 1, -1]
        )

        cos = gpytorch.kernels.CosineKernel()
        cos.period_length = math.pi

        lp, lc = get_prior_and_constraint(lengthscale_prior,
                                          lengthscale_constraint)
        matern_1 = gpytorch.kernels.MaternKernel(
            nu=2.5,
            ard_num_dims=1,
            lengthscale_prior=lp,
            lengthscale_constraint=lc
        )
        l_init = get_initialization(
            'matern_1.lengthscale', hyperparameters_initialization, lp
        )
        matern_1.lengthscale = l_init
        sym_matern_1 = matern_1 + ConjugateKernel(matern_1, conjugation=[-1])

        prod_decomp = ProductDecompositionKernel(
            (sym_matern_0, 4),
            (cos, 1),
            (sym_matern_1, 1)
        )
        op, oc = get_prior_and_constraint(outputscale_prior,
                                          outputscale_constraint)

        covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel=prod_decomp,
            outputscale_prior=op,
            outputscale_constraint=oc
        )
        o_init = get_initialization(
            'outputscale', hyperparameters_initialization, op
        )
        covar_module.outputscale = o_init

        noise_p, nc = get_prior_and_constraint(noise_prior,
                                          noise_constraint)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=noise_p,
            noise_constraint=nc
        )
        n_init = get_initialization(
            'noise_covar.noise', hyperparameters_initialization, op
        )
        likelihood.noise_covar.noise = n_init

        super(SymmetricMaternCosGP, self).__init__(train_x, train_y,
                                                   mean_module, covar_module,
                                                   likelihood, **kwargs)

    @property
    def structure_dict(self):
        return self.__structure_dict