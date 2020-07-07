from itertools import product
import numpy as np

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

    @property
    def limits(self):
        raise NotImplementedError

    @staticmethod
    def element(*x):
        if len(x) == 1:
            return np.atleast_1d(x[0])
        else:
            return np.atleast_1d(x)


class DiscretizableSpace(Space):
    def __init__(self, index_shape):
        super(DiscretizableSpace, self).__init__()

        self.index_shape = index_shape
        self.index_dim = len(self.index_shape)
        self.data_length = self.index_dim

    @property
    def shape(self):
        return self.index_shape

    def contains(self, x):
        raise NotImplementedError

    def is_on_grid(self, x):
        raise NotImplementedError

    def get_index_of(self, x, around_ok=False):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def sample_idx(self):
        ids = tuple(map(np.random.choice, self.index_shape))
        if len(ids) == 1:
            return ids[0]
        else:
            return ids

    def sample(self):
        k = self.sample_idx()
        return self[k]

    def __iter__(self):
        return DiscretizableSpaceIterator(self)


class DiscretizableSpaceIterator:
    def __init__(self, space):
        self.space = space
        if space.index_dim > 1:
            self.index_iter = product(
                *[range(l_k) for l_k in space.index_shape]
            )
        else:
            self.index_iter = iter(range(space.index_shape[0]))

    def __iter__(self):
        return self

    def __next__(self):
        index = next(self.index_iter)
        data = self.space[index]
        return (index, data)


class ProductSpace(DiscretizableSpace):
    def __init__(self, *sets):
        self._flattened_sets = []
        for s in sets:
            if isinstance(s, ProductSpace):
                for second_order_set in s._flattened_sets:
                    self._flattened_sets.append(second_order_set)
            else:
                self._flattened_sets.append(s)
        self._n_flattened_sets = len(self._flattened_sets)

        index_shape = tuple([s.index_shape[0] for s in self._flattened_sets])

        super(ProductSpace, self).__init__(index_shape)

        self.sets = sets
        self.n_sets = len(sets)

        self._index_slices = [None] * self.n_sets
        current_index = 0
        for ns in range(self.n_sets):
            end_index = current_index + sets[ns].index_dim
            self._index_slices[ns] = slice(current_index, end_index)
            current_index = end_index

    def __getitem__(self, index):
        if isinstance(index, np.ndarray):
            if index in self:
                return index
            else:
                index in self
                raise IndexError(f'Index {index} is understood as an element '
                                 'of the Space and does not belong to it')

        if not isinstance(index, tuple):
            index = tuple([index])

        n_missing_indexes = self._n_flattened_sets - len(index)
        index = index + tuple([slice(None, None, None)
                               for k in range(n_missing_indexes)])

        def get_dim(ns):
            return np.atleast_1d(self._flattened_sets[ns][index[ns]])

        def isnotslice(x):
            return not isinstance(x, slice)

        list_of_items = list(map(get_dim, list(range(self._n_flattened_sets))))
        item_is_1d = list(map(isnotslice, index))

        # NumPy limits the dimension of arrays to 32, so we need to be careful when meshgridding, and only extend
        # the dimensions along which the user has asked for more than 1 value (i.e., a slice)
        items_multidimensional = [item for item, is_1d in zip(list_of_items, item_is_1d) if not is_1d]
        if len(items_multidimensional) > 0:
            items_multidimensional_meshgrid = np.meshgrid(*items_multidimensional, indexing='ij')
            items_shape = items_multidimensional_meshgrid[0].shape
            idx_in_multidim = 0
            for item_index in range(len(list_of_items)):
                if not item_is_1d[item_index]:
                    list_of_items[item_index] = items_multidimensional_meshgrid[idx_in_multidim]
                    idx_in_multidim += 1
                else:
                    assert list_of_items[item_index].shape == (1,)
                    value = list_of_items[item_index][0]
                    list_of_items[item_index] = value * np.ones(items_shape)
            items = np.stack(list_of_items, axis=-1)
        else:
            # squeeze returns a np scalar if the input is of shape (1,), so we ensure it is still an array
            items = np.atleast_1d(np.stack(list_of_items, axis=0).squeeze())

        # squeeze_dim = list(map(isnotslice, index))
        #
        # items_meshgrid = np.meshgrid(*list_of_items, indexing='ij')
        # items = np.stack(items_meshgrid, axis=-1)
        #
        # dims_to_squeeze = tuple([dim
        #                         for dim in range(len(squeeze_dim))
        #                         if squeeze_dim[dim]])
        # items = np.squeeze(items, axis=dims_to_squeeze)
        return items

    def _get_components(self, x, ns):
        return x[self._index_slices[ns]]

    def contains(self, x):
        if len(x) != self.data_length:
            raise ValueError(f"Size mismatch: expected size {self.data_length}"
                             f", got {len(x)}")
        isin = True
        for ns in range(self.n_sets):
            s = self.sets[ns]
            x_slice = self._get_components(x, ns)
            isin = isin and (x_slice in s)

        return isin

    def is_on_grid(self, x):
        if len(x) != self.data_length:
            raise ValueError(f"Size mismatch: expected size {self.data_length}"
                             f", got {len(x)}")
        ison = True
        for ns in range(self.n_sets):
            s = self.sets[ns]
            x_slice = self._get_components(x, ns)
            ison = ison and s.is_on_grid(x_slice)

        return ison

    def get_index_of(self, x, around_ok=False):
        if len(x) != self.data_length:
            raise ValueError(f'Size mismatch: expected size {self.data_length}'
                             f', got {len(x)}')
        index = [None] * self.index_dim
        for ns in range(self.n_sets):
            s = self.sets[ns]
            index_slice = self._index_slices[ns]
            # It is only possible to assign iterables to slices, so we need
            # the ensure_list wrapping
            index[index_slice] = np.atleast_1d(
                s.get_index_of(x[index_slice], around_ok)
            ).tolist()
        return tuple(index)

    def closest_in(self, x):
        if x in self:
            return x
        y = x.copy()
        for ns in range(self.n_sets):
            mask = self._index_slices[ns]
            y[mask] = self.sets[ns].closest_in(x[mask])
        return y

    @property
    def limits(self):
        limits = [None] * self._n_flattened_sets
        for ns in range(self._n_flattened_sets):
            s = self._flattened_sets[ns]
            limits[ns] = s.limits
        return tuple(limits)


    def get_component(self, x, target):
        if target not in self.sets:
            raise error.InvalidTarget
        n_target = None
        for ns in range(self.n_sets):
            if self.sets[ns] == target:
                n_target = ns
                break
        else:
            raise error.InvalidTarget
        mask = self._index_slices[n_target]
        return np.hstack(x[mask])

    def get_index_component(self, index, target):
        if target not in self.sets:
            raise error.InvalidTarget
        n_target = None
        for ns in range(self.n_sets):
            if self.sets[ns] == target:
                n_target = ns
                break
        else:
            raise error.InvalidTarget
        mask = self._index_slices[n_target]
        masked = index[mask]
        if len(masked) == 1:
            return masked[0]
        return masked

    def from_components(self, *x_sets):
        return np.hstack(x_sets)
