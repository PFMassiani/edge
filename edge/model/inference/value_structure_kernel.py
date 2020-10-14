import functools
from gpytorch.models.exact_prediction_strategies import \
    DefaultPredictionStrategy, clear_cache_hook
from gpytorch.utils.memoize import cached
from gpytorch import settings, delazify
from gpytorch.kernels import Kernel
from gpytorch.means import Mean
from gpytorch.lazy import DiagLazyTensor, ZeroLazyTensor, CatLazyTensor, \
    LazyEvaluatedKernelTensor
from torch import Size, logical_not
from .tensorwrap import ensure_tensor
from edge.utils import device


class DiscountedPredictionStrategy(DefaultPredictionStrategy):
    def __init__(self, train_inputs, train_prior_dist, train_labels, likelihood,
                 discount_tensor, is_terminal, root=None, inv_root=None):
        # train_prior_dist is the output of gp(gp.train_inputs)
        # -> (MultivariateNormal)Distribution
        super(DiscountedPredictionStrategy, self).__init__(
            train_inputs, train_prior_dist, train_labels, likelihood, root,
            inv_root
        )
        self.discount_tensor = discount_tensor
        self.is_terminal = is_terminal

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
    def __init__(self, base_kernel, discount_factor, dataset, **kwargs):
        if base_kernel.active_dims is not None:
            kwargs["active_dims"] = base_kernel.active_dims
        super(ValueStructureKernel, self).__init__(**kwargs)
        self.base_kernel = base_kernel
        self.discount_factor = discount_factor
        self.dataset = dataset

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels,
                            likelihood):
        return DiscountedPredictionStrategy(
            train_inputs=train_inputs,
            train_prior_dist=train_prior_dist,
            train_labels=train_labels,
            discount_tensor=self._discount_tensor(train_inputs),
            is_terminal=self.dataset.is_terminal,
            likelihood=likelihood,
        )

    def __call__(self, x1, x2=None, diag=False, last_dim_is_batch=False, **params):
        res = super(ValueStructureKernel, self).__call__(
            x1, x2, diag, last_dim_is_batch, **params
        )
        if not diag and settings.lazily_evaluate_kernels.on():
            res = LazyEvaluatedValueStructureKernelTensor(
                x1=res.x1, x2=res.x2, kernel=res.kernel,
                last_dim_is_batch=res.last_dim_is_batch,
                kernel_in_training=self.training, **res.params
            )
        return res

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        res = self.base_kernel.forward(
            x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params
        )
        if self.training:
            # In this case, the prediction strategy is not called in the
            # calling class: we need to do it here
            discount_tensor = self._discount_tensor(t=x1.shape[0]).evaluate()
            res = discount_tensor.matmul(
                res.matmul(
                    discount_tensor.transpose(-1, -2)
                )
            )
            return res
        else:
            return res

    def _discount_tensor(self, train_inputs=None, t=None):
        tm1 = len(train_inputs[0]) - 1 if t is None else t - 1

        eye_tm1 = DiagLazyTensor(ensure_tensor([1.] * tm1))
        gamma_tm1 = DiagLazyTensor(
            ensure_tensor([- self.discount_factor] * tm1)
        )
        if self.dataset.has_is_terminal:
            terminal_filter = DiagLazyTensor(
                logical_not(self.dataset.is_terminal[:tm1])
            )
            gamma_tm1 = terminal_filter.matmul(gamma_tm1)

        # TODO make this more general by replacing the 1 with num_tasks
        diag_part = CatLazyTensor(
            eye_tm1, ZeroLazyTensor(tm1, 1, device=device), dim=-1,
            output_device=device
        )
        superdiag_part = CatLazyTensor(
            ZeroLazyTensor(tm1, 1, device=device), gamma_tm1, dim=-1,
            output_device=device
        )
        discount_tensor = diag_part + superdiag_part
        return discount_tensor


class LazyEvaluatedValueStructureKernelTensor(LazyEvaluatedKernelTensor):
    def __init__(self, x1, x2, kernel, last_dim_is_batch=False,
                 kernel_in_training=False, **params):
        super(LazyEvaluatedValueStructureKernelTensor, self).__init__(
            x1, x2, kernel, last_dim_is_batch, **params
        )
        self.kernel_in_training = kernel_in_training

    @LazyEvaluatedKernelTensor.shape.getter
    def shape(self):
        size = self.size()
        if self.kernel_in_training:
            dim_data = -2 if self.last_dim_is_batch else -1
            end_append = size[-1:] if self.last_dim_is_batch else tuple()
            size = size[:dim_data - 1] + (
                size[dim_data - 1] - 1, size[dim_data] - 1
            ) + end_append
        return size


class ValueStructureMean(Mean):
    def __init__(self, base_mean):
        super(ValueStructureMean, self).__init__()
        self.base_mean = base_mean

    def forward(self, x):
        res = self.base_mean.forward(x)
        if self.training:
            res = res[:-1]
        return res
