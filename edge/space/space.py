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

    def closest_in(self, x):
        raise NotImplementedError


class DiscretizableSpace(Space):
    def __init__(self, discretization):
        super(DiscretizableSpace, self).__init__()

        self.discretization = discretization
        self.index_shape = self.discretization.shape[:-1]
        self.index_dim = len(self.index_shape)
        self.data_length = self.discretization.shape[-1]

    def _get_index_with_data_component(self, index):
        if len(index) == self.index_dim + 1:
            raise IndexError(f'Expected length {self.index_dim} for index, got'
                             f'{len(index)}. The last dimension is used for th'
                             'e data, and should not be used for indexing on a'
                             ' Space object. Try directly indexing on '
                             'Space.discretization instead')
        elif len(index) != self.index_dim:
            raise IndexError(f'Expected length {self.index_dim} for index, got'
                             f'{len(index)}')
        return tuple(ensure_list(index)) + \
            tuple(ensure_list(slice(None, None, None)))

    def contains(self, x):
        raise NotImplementedError

    def is_on_grid(self, x):
        raise NotImplementedError

    def __getitem__(self, index):
        index = self._get_index_with_data_component(index)
        return self.discretization[index]

    def __iter__(self):
        return DiscretizableSpaceIterator(self)

    def get_index_of(self, x, around_ok=False):
        raise NotImplementedError

    def sample_idx(self):
        ids = tuple(map(np.random.choice, self.index_shape))
        return ids

    def sample(self):
        k = self.sample_idx()
        return self[k]


class DiscretizableSpaceIterator:
    def __init__(self, space):
        self.space = space
        self.index_iter = product(
            *[range(l_k) for l_k in space.index_shape]
        )

    def __iter__(self):
        return self

    def __next__(self):
        index = next(self.index_iter)
        data = self.space[index]
        return (index, data)


class ProductSpace(DiscretizableSpace):
    def __init__(self, *sets):
        self.sets = sets
        self.n_sets = len(self.sets)
        grids_to_mesh = [None] * self.n_sets
        is_product = [False] * self.n_sets
        for ns in range(self.n_sets):
            if isinstance(sets[ns], ProductSpace):
                grids_to_mesh[ns] = sets[ns].discretization_grid
                is_product[ns] = True
            else:
                grids_to_mesh[ns] = sets[ns].discretization

        discretization_grids = []
        for subgrids, isprod in zip(grids_to_mesh, is_product):
            if isprod:
                for grid in subgrids:
                    discretization_grids.append(grid)
            else:
                discretization_grids.append(subgrids.reshape(-1))

        mesh = np.meshgrid(discretization_grids, indexing='ij')
        discretization = np.stack(mesh, axis=-1)

        super(ProductSpace, self).__init__(discretization)

        self._index_slices = [None] * self.n_sets
        current_index = 0
        for ns in range(self.n_sets):
            end_index = current_index + self.sets[ns].index_dim
            self._index_slices[ns] = slice(current_index, end_index)
            current_index = end_index

    def _get_components(self, x, ns):
        return x[self._index_slices[ns]]

    def contains(self, x):
        if len(x) != self.data_length:
            raise ValueError(f"Size mismatch: expected size {self.data_length}"
                             f", got {len(x)}")
        isin = True
        for ns in range(self._n_sets):
            s = self.sets[ns]
            x_slice = self._get_components(x, ns)
            isin = isin and (x_slice in s)

        return isin

    def is_on_grid(self, x):
        if len(x) != self.data_length:
            raise ValueError(f"Size mismatch: expected size {self.data_length}"
                             f", got {len(x)}")
        ison = True
        for ns in range(self._n_sets):
            s = self.sets[ns]
            x_slice = self._get_components(x, ns)
            ison = ison and s.is_on_grid(x_slice)

        return ison

    def get_index_of(self, x, around_ok=False):
        if len(x) != self.data_length:
            raise ValueError(f'Size mismatch: expected size {self.data_length}'
                             f', got {len(x)}')
        index = np.zeros(self.index_dim, dtype=np.int)
        for ns in range(self._n_sets):
            s = self.sets[ns]
            mask = self._index_masks[ns]
            index[mask] = s.get_index_of(x[mask], around_ok)
        return index

    def closest_in(self, x):
        if x in self:
            return x
        y = np.array_like(x)
        for ns in range(self._n_sets):
            mask = self._index_masks[ns]
            y[mask] = self.sets[ns].closest_in(x[mask])
        return y

    def get_component(self, x, target):
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
        return np.atleast_1d(x[mask])

    def from_components(self, *x_sets):
        return np.hstack(x_sets)
