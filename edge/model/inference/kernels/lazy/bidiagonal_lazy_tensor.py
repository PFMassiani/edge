import torch
from gpytorch.lazy import LazyTensor
from gpytorch.utils.memoize import cached


def upper_bidiagonal_pseudoinverse(off_diag, square):
    if off_diag.ndimension() > 1:
        return [upper_bidiagonal_pseudoinverse(batch, square)
                for batch in off_diag]
    else:
        ncols = off_diag.size(-1)
        nrows = ncols if square else ncols + 1
        return [[
            (-off_diag[i:j]).prod() if i < j else 1 if i == j else 0
            for j in range(ncols)
        ] for i in range(nrows)]


class BidiagonalLazyTensor(LazyTensor):
    def __init__(self, off_diag, upper=True, square=True):
        """
        Bidiagonal lazy tensor with ones on the diagonal
        Args:
            :attr:`off_diag` (Tensor or LazyTensor):
                A `b1 x ... x bk x n` Tensor, representing a `b1 x ... x bk`-sized
                batch of `(n+1) x (n+1)`, `n x (n+1)` or `(n+1) x n` bidiagonal
                matrices, depending on the values of `upper` and `square`
            :attr: `upper` (bool):
                If True, the Tensor is constructed to be upper-bidiagonal,
                otherwise lower-bidiagonal
            :attr: `square` (bool):
                If True, the Tensor is a square matrix with ones on the
                diagonal. Otherwise, the last row (resp. column if upper=False)
                is removed.
        """
        super().__init__(off_diag, upper=upper, square=square)
        self._off_diag = off_diag
        self.upper = upper
        self.square = square

    def _size(self):
        n = self._off_diag.size(-1) + 1  # Size of square matrix is size of superdiagonal + 1
        m = n if self.square else n - 1  # Remove last row if not square
        if not self.upper:
            m, n = n, m
        return self._off_diag.shape[:-1] + torch.Size([m, n])

    def _transpose_nonbatch(self):
        return BidiagonalLazyTensor(self._off_diag, not self.upper, self.square)

    def _matmul(self, rhs):
        # We decompose the result in two parts: diag_res and off_diag_res
        # diag_res is the result of eye(p) * rhs, where p = self.size(-1) + 1 or
        #   self.size(-1) depending on self.square and rhs is masked accordingly
        # off_diag_res is the result of diag(self._off_diag) * rhs, where the
        #   first or last row of rhs is removed, and possibly concatenated with
        #   a row of 0 at the first or last row depending on self.square
        from gpytorch.lazy import DiagLazyTensor
        is_vector = rhs.ndimension() == 1
        if is_vector:
            rhs = rhs.unsqueeze(-1)

        alldim = slice(None, None, None)  # Alias to shorten code
        batch_size = rhs.shape[:-2]
        batch_slices = tuple(
            alldim for _ in range(len(batch_size))
        )

        # Off diag
        extract_off_diag = (slice(1,  None, None), alldim) if self.upper else\
                           (slice(None, -1, None), alldim) if self.square else\
                           (alldim, alldim)
        extract_off_diag = batch_slices + extract_off_diag
        off_diag_rhs = rhs[extract_off_diag]
        off_diag_res = DiagLazyTensor(self._off_diag).matmul(off_diag_rhs)
        if (not self.square) and self.upper:
            pass
        else:
            zero_row = torch.zeros(
                *batch_size, 1, rhs.size(-1), dtype=rhs.dtype, device=rhs.device
            )
            to_cat = (off_diag_res, zero_row) if self.upper else \
                     (zero_row, off_diag_res)
            off_diag_res = torch.cat(to_cat, dim=-2)

        # Diag
        if self.square:
            diag_res = rhs
        elif self.upper:
            extract_diag = batch_slices + (slice(None, -1, None), alldim)
            diag_res = rhs[extract_diag]
        else:
            zero_row = torch.zeros(
                *batch_size, 1, rhs.size(-1), dtype=rhs.dtype, device=rhs.device
            )
            diag_res = torch.cat((rhs, zero_row), dim=-2)

        res = diag_res + off_diag_res
        if is_vector:
            res = res.squeeze(-1)
        return res

    def _solve(self, rhs, preconditioner, num_tridiag=0):
        is_vector = rhs.ndimension() == 1
        if is_vector:
            rhs = rhs.unsqueeze(-1)
        pseudoinverse = upper_bidiagonal_pseudoinverse(self._off_diag,
                                                       self.square)
        pseudoinverse = torch.tensor(
            pseudoinverse, dtype=self.dtype, device=rhs.device
        )
        if not self.upper:
            pseudoinverse = pseudoinverse.transpose(-1, -2)
        res = pseudoinverse.matmul(rhs)
        if is_vector:
            res = res.squeeze(-1)
        return res

    def _get_indices(self, row_index, col_index, *batch_indices):
        batch_lens = tuple(len(indices) for indices in batch_indices)
        nrows = row_index.numel()
        ncols = col_index.numel()
        target_shape = [max(rdim, cdim)
                        for rdim, cdim in zip(row_index.shape, col_index.shape)]
        diag_res = torch.ones(
            *batch_lens, *target_shape, dtype=self.dtype, device=self.device
        )
        if self.upper:
            off_diag_res = self._off_diag[row_index].flatten().expand(
                *batch_lens, ncols, nrows
            ).clone().transpose(-1, -2)
        else:
            off_diag_res = self._off_diag[col_index].flatten().expand(
                *batch_lens, nrows, ncols
            ).clone()
        off_diag_res = off_diag_res.reshape(target_shape)
 
        sign = 1 if self.upper else -1
        diag_res = diag_res * torch.eq(row_index, col_index).to(
            device=self.device, dtype=self.dtype
        )
        off_diag_res = off_diag_res * torch.eq(row_index + sign, col_index).to(
            device=self.device, dtype=self.dtype
        )
        return diag_res + off_diag_res

    def left_matmul(self, other):
        return self.transpose(-1, -2).matmul(
            other.transpose(-1, -2)
        ).transpose(-1, -2)

    @cached
    def evaluate(self):
        n = self._off_diag.size(-1)
        off_diag = torch.cat(
            (
                torch.zeros(n, 1, dtype=self.dtype, device=self.device),
                torch.diag(self._off_diag)
            ),
            dim=-1
        )

        p = n if not self.square else n + 1
        diag = torch.tensor([1] * p, dtype=self.dtype, device=self.device)
        diag = torch.diag(diag)
        if not self.square:
            diag = torch.cat(
                (diag, torch.zeros(n, 1, dtype=self.dtype, device=self.device)),
                dim=-1
            )

        res = diag + off_diag
        if not self.upper:
            res = res.transpose(-1, -2)
        return res


__all__ = ['BidiagonalLazyTensor']
