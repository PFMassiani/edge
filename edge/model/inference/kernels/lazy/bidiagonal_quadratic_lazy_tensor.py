from gpytorch.lazy import MatmulLazyTensor, NonLazyTensor, LazyTensor


class BidiagonalQuadraticLazyTensor(MatmulLazyTensor):
    def __init__(self, bidiagonal_tensor, center_tensor):
        left_lazy_tensor = bidiagonal_tensor.matmul(center_tensor)
        right_lazy_tensor = bidiagonal_tensor.transpose(-1, -2)
        super().__init__(left_lazy_tensor, right_lazy_tensor)
        # We call the constructor of LazyTensor to redefine _args and __kwargs
        LazyTensor.__init__(self, bidiagonal_tensor, center_tensor)

        self.bidiagonal_tensor = bidiagonal_tensor
        self.center_tensor = center_tensor

    def diag(self):
        # Forces the evaluation of self to compute the diagonal
        # This enables using the structure of the Tensor implemented in _matmul,
        # and avoids multiplying tensors in the _get_indices in O(n^3)
        return NonLazyTensor(self.evaluate()).diag()
