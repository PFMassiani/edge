import torch
from gpytorch.kernels import Kernel, ProductKernel
from .tensorwrap import ensure_tensor
from edge.utils import device


def get_index_and_name(name):
    split_name = name.split('.')
    index = int(split_name[0])
    pname = '.'.join(split_name[1:])
    return index, pname


# TODO is conjugating the input okay ?
class ConjugateKernel(Kernel):
    def __init__(self, base_kernel, conjugation):
        super(ConjugateKernel, self).__init__()
        self.base_kernel = base_kernel
        self.conjugation = ensure_tensor(conjugation)

    @property
    def is_stationary(self) -> bool:
        return self.base_kernel.is_stationary

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return self.base_kernel.prediction_strategy(
            train_inputs, train_prior_dist, train_labels, likelihood
        )

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        x2_ = x2 * self.conjugation
        return self.base_kernel(x1, x2_, diag=diag,
                                last_dim_is_batch=last_dim_is_batch, **params)


class ProductDecompositionKernel(ProductKernel):
    def __init__(self, *kernel_and_n_dims):
        self.base_kernels, n_dims = list(zip(*kernel_and_n_dims))
        super(ProductDecompositionKernel, self).__init__(*self.base_kernels)
        n_kernels = len(self.base_kernels)
        total_dims = sum(n_dims)
        self.masks = [
            torch.zeros(total_dims, dtype=bool, device=device)
            for _ in range(n_kernels)
        ]
        start = 0
        for n in range(n_kernels):
            end = start + n_dims[n]
            self.masks[n][start:end] = True
            start = end

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("ProductDecompositionKernel does not accept the "
                               "last_dim_is_batch argument.")
        slice_dims = x1.ndimension() - 1
        mask_first_dims = tuple(slice(None, None, None)
                                for _ in range(slice_dims))
        masks = [mask_first_dims + (m,) for m in self.masks]
        res = None
        for kernel, mask in zip(self.base_kernels, masks):
            next_term = kernel.forward(x1[mask], x2[mask], diag=diag,
                                       last_dim_is_batch=False, **params)
            if res is None:
                res = next_term
            else:
                res = res * next_term
        return res

    def initialize(self, **kwargs):
        for name, val in kwargs.items():
            index, pname = get_index_and_name(name)
            self.base_kernels[index].initialize(**{pname: val})
