import functools
from gpytorch.models.exact_prediction_strategies import \
    DefaultPredictionStrategy, clear_cache_hook
from gpytorch.utils.memoize import cached
from gpytorch import settings, delazify
from gpytorch.kernels import Kernel
from gpytorch.lazy import DiagLazyTensor, ZeroLazyTensor, CatLazyTensor
from torch import tensor, Size


class DiscountedPredictionStrategy(DefaultPredictionStrategy):
    def __init__(self, train_inputs, train_prior_dist, train_labels, likelihood,
                 discount_tensor, root=None, inv_root=None):
        # train_prior_dist is the output of gp(gp.train_inputs)
        # -> (MultivariateNormal)Distribution
        super(DiscountedPredictionStrategy, self).__init__(
            train_inputs, train_prior_dist, train_labels, likelihood, root,
            inv_root
        )
        self.discount_tensor = discount_tensor

    @property
    @cached(name="mean_cache")
    def mean_cache(self):
        # Code adapted from the source of
        # gpytorch.models.exact_prediction_strategies.DefaultPredictionStrategy
        # This function is used in self.exact_predictive_mean

        # The covariance matrix without noise is accessed with
        # self.train_prior_dist.lazy_covariance_matrix
        train_train_nonoise_covar = self.train_prior_dist.lazy_covariance_matrix

        # We can only use the labels up to t-1
        inputs_shape = self.train_prior_dist.mean.shape
        tm1_noise_shape = Size(
            inputs_shape[:-1] + (inputs_shape[-1] - 1,)
        )
        # This part is adapted from GaussianLikelihood.marginal. The difference
        # is because we can only use the labels up to t-1
        noise_covar = self.likelihood._shaped_noise_covar(
            base_shape=tm1_noise_shape
        )
        mvn = self.likelihood(self.train_prior_dist, self.train_inputs)
        train_mean = mvn.loc
        # TODO error when evaluating self.discount_tensor
        train_train_covar = self.discount_tensor.matmul(
            train_train_nonoise_covar.matmul(
                self.discount_tensor.transpose(-1, -2)
            )
        ) + noise_covar

        # Careful! Possible source of error. The mean is computed using labels
        # up to t, but we only use the ones up to t-1 later -> Check this is ok
        train_labels_offset = (self.train_labels - train_mean).unsqueeze(-1)
        # We can only use the information up to t-1
        train_labels_offset = train_labels_offset[:-1]
        mean_cache = self.discount_tensor.transpose(-1, -2).matmul(
            train_train_covar.inv_matmul(train_labels_offset)
        ).squeeze(-1)

        if settings.detach_test_caches.on():
            mean_cache = mean_cache.detach()

        if mean_cache.grad_fn is not None:
            wrapper = functools.partial(clear_cache_hook, self)
            functools.update_wrapper(wrapper, clear_cache_hook)
            mean_cache.grad_fn.register_hook(wrapper)

        return mean_cache

    @property
    @cached(name="covar_cache")
    def covar_cache(self):
        # Code adapted from the source of
        # gpytorch.models.exact_prediction_strategies.DefaultPredictionStrategy

        inputs_shape = self.train_prior_dist.mean.shape
        tm1_noise_shape = Size(
            inputs_shape[:-1] + (inputs_shape[-1] - 1,)
        )
        noise_covar = self.likelihood._shaped_noise_covar(
            base_shape=tm1_noise_shape
        )
        train_train_nonoise_covar = self.train_prior_dist.lazy_covariance_matrix
        train_train_covar = self.discount_tensor.matmul(
            train_train_nonoise_covar.matmul(
                self.discount_tensor.transpose(-1, -2)
            )
        ) + noise_covar

        train_train_covar_inv_root = delazify(
            self.discount_tensor.transpose(-1, -2).matmul(
                train_train_covar.root_inv_decomposition().root
            )
        )
        return self._exact_predictive_covar_inv_quad_form_cache(
            train_train_covar_inv_root, self._last_test_train_covar)


class ValueStructureKernel(Kernel):
    def __init__(self, base_kernel, discount_factor, **kwargs):
        if base_kernel.active_dims is not None:
            kwargs["active_dims"] = base_kernel.active_dims
        super(ValueStructureKernel, self).__init__(**kwargs)
        self.base_kernel = base_kernel
        self.discount_factor = discount_factor

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels,
                            likelihood):
        return DiscountedPredictionStrategy(
            train_inputs=train_inputs,
            train_prior_dist=train_prior_dist,
            train_labels=train_labels,
            discount_tensor=self._discount_tensor(train_inputs),
            likelihood=likelihood,
        )

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        return self.base_kernel.forward(
            x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params
        )

    def _discount_tensor(self, train_inputs):
        tm1 = len(train_inputs[0]) - 1

        eye_tm1 = DiagLazyTensor(tensor([1.] * tm1))
        gamma_tm1 = DiagLazyTensor(tensor([- self.discount_factor] * tm1))

        # TODO make this more general by replacing the 1 with num_tasks
        diag_part = CatLazyTensor(
            eye_tm1, ZeroLazyTensor(tm1, 1), dim=-1
        )
        superdiag_part = CatLazyTensor(
            ZeroLazyTensor(tm1, 1), gamma_tm1, dim=-1
        )
        discount_tensor = diag_part + superdiag_part
        return discount_tensor


def add_value_structure(gp, discount_factor):
    gp.covar_module = ValueStructureKernel(
        base_kernel=gp.covar_module,
        discount_factor=discount_factor,
    )
    gp.structure_dict["value_structure"] = {"discount_factor": discount_factor}