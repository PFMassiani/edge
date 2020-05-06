from itertools import product, starmap
import numpy as np

from edge.utils import ensure_list
from edge import error


class Space:
    def __init__(self):
        pass

    def contains(self, x):
        raise NotImplementedError

    def __contains__(self, x):
        return self.contains(x)

    def sample(self):
        raise NotImplementedError


class DiscretizableSpace(Space):
    def __init__(self, index_dim, discretization_shape):
        super(DiscretizableSpace, self).__init__()
        discretization_shape = np.atleast_1d(discretization_shape)
        if len(discretization_shape) != index_dim:
            raise IndexError('Size mismatch: expected shape of discretization '
                             f'grid to be {index_dim}, got '
                             f'{len(discretization_shape)} instead')
        self.index_dim = index_dim
        self.discretization_shape = discretization_shape

    def contains(self, x):
        raise NotImplementedError

    def is_on_grid(self, x):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __iter__(self):
        return self.get_index_iterator()

    def get_index_iterator(self):
        raise NotImplementedError

    def get_index_of(self, x, around_ok=False):
        raise NotImplementedError

    def sample_idx(self):
        raise NotImplementedError

    def sample(self):
        k = self.sample_idx()
        return self[k]


class ProductSpace(DiscretizableSpace):
    def __init__(self, *sets):
        self.sets = sets
        self._n_sets = len(self.sets)
        discretization_shape = [0
                                for ns in range(self._n_sets)
                                for k in range(sets[ns].index_dim)
                                ]
        index_dim = 0
        for s in self.sets:
            s_end = index_dim + s.index_dim
            discretization_shape[index_dim:s_end] = s.discretization_shape
            index_dim = s_end

        super(ProductSpace, self).__init__(index_dim, discretization_shape)

        self._index_masks = [None] * self._n_sets
        current_index = 0
        for ns in range(self._n_sets):
            end_index = current_index + self.sets[ns].index_dim
            self._index_masks[ns] = slice(current_index, end_index)
            current_index = end_index

    def contains(self, x):
        if len(x) != self.index_dim:
            raise ValueError(f"Size mismatch: expected size {self.index_dim}"
                             f", got {len(x)}")
        isin = True
        for ns in range(self._n_sets):
            s = self.sets[ns]
            mask = self._index_masks[ns]
            isin = isin and (x[mask] in s)

        return isin

    def is_on_grid(self, x):
        if len(x) != self.index_dim:
            raise ValueError(f"Size mismatch: expected size {self.index_dim}"
                             f", got {len(x)}")
        ison = True
        for ns in range(self._n_sets):
            s = self.sets[ns]
            mask = self._index_masks[ns]
            ison = ison and s.is_on_grid(x[mask])

        return ison

    def __getitem__(self, index):
        if len(index) != self.index_dim:
            raise ValueError(f'Size mismatch: expected size {len(self.sets)}'
                             f', got {len(index)}')
        x = []
        for ns in range(self._n_sets):
            mask = self._index_masks[ns]
            index_in_set = index[mask]
            s = self.sets[ns]
            x = np.concatenate((x, s[index_in_set]))
        return x

    def get_index_of(self, x, around_ok=False):
        if len(x) != self.index_dim:
            raise ValueError(f'Size mismatch: expected size {len(self.sets)}'
                             f', got {len(x)}')
        index = np.zeros(self.index_dim, dtype=np.int)
        for ns in range(self._n_sets):
            s = self.sets[ns]
            mask = self._index_masks[ns]
            index[mask] = s.get_index_of(x[mask], around_ok)
        return index

    def get_index_iterator(self):
        def flatten(*args):
            x = []
            for y in args:
                z = ensure_list(y)
                x += z
            return np.array(x, dtype=np.int)
        return starmap(
            flatten,
            product(*self.sets)
        )

    def sample_idx(self):
        def sample_set(s):
            index = s.sample_idx()
            index = np.array(ensure_list(index), dtype=np.int).reshape(-1)
            return index
        return np.concatenate(tuple(map(sample_set, self.sets)))

    def get_projection_on_space(self, x, target):
        if target not in self.sets:
            raise error.InvalidTarget
        n_target = None
        for ns in range(self._n_sets):
            if self.sets[ns] == target:
                n_target = ns
                break
        else:
            raise error.InvalidTarget
        mask = self._index_masks[n_target]
        return x[mask]
