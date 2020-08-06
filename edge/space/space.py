from itertools import product
import numpy as np

from edge import error


class Space:
    """Base data structure to handle state, action, and stateaction spaces
    A Space is an object that contains elements, from where we can sample, and that has limits
    """
    def __init__(self):
        pass

    def contains(self, x):
        """
        Abstract method.
        :param x:
        :return: True iff x is in the Space
        """
        raise NotImplementedError

    def __contains__(self, x):
        return self.contains(x)

    def sample(self):
        """Abstract method.
        Samples an element from the Space
        """
        raise NotImplementedError

    def closest_in(self, x):
        """ Abstract method
        Returns the closest element in the Space
        :param x:
        :return:
        """
        raise NotImplementedError

    @property
    def limits(self):
        """Returns the limits of the Space
        :return: tuple<float>
        """
        raise NotImplementedError

    @staticmethod
    def element(*x):
        """
        Wraps the arguments with the data structure used by Space elements, i.e., a np.ndarray. Approximately
        equivalent to np.atleast_1d.
        :param x: The list of arguments
        :return: element: np.ndarray
        """
        if len(x) == 1:
            return np.atleast_1d(x[0])
        else:
            return np.atleast_1d(x)


class DiscretizableSpace(Space):
    """
    Data structure to handle discretizable Spaces. The main difference is that a DiscretizableSpace can be indexed.
    The discretization is only useful when indexing the space, and not when checking that something is an element of
    the space
    """
    def __init__(self, index_shape):
        """
        :param index_shape: tuple: the shape of the discretization. This would correspond to the shape of the numpy
            array if you used it instead of a DiscretizableSpace
        """
        super(DiscretizableSpace, self).__init__()

        self.index_shape = index_shape
        self.index_dim = len(self.index_shape)
        # DiscretizableSpaces are more complex than np.ndarrays. Indeed, an additional dimension is created after the
        # last indexing dimension. Then, the space can be seen as a np.ndarray of dimension `index_dim` whose values
        # are np.ndarrays of shape `(data_length,)`
        self.data_length = self.index_dim

    @property
    def shape(self):
        return self.index_shape

    def contains(self, x):
        raise NotImplementedError

    def is_on_grid(self, x):
        """ Abstract method
        :param x: the element of the space
        :return: boolean: True iff the element is on the discretization grid
        """
        raise NotImplementedError

    def get_index_of(self, x, around_ok=False):
        """ Abstract method
        Returns the index of an element
        :param x: the element
        :param around_ok: boolean: whether the element should be exactly on the grid (False) or if some tolerance
            is accepted (True)
        :return: boolean
        """
        raise NotImplementedError

    def __getitem__(self, index):
        """ Abstract method
        The indexing method
        :param index: the index
        :return: np.ndarray : the item
        """
        raise NotImplementedError

    def sample_idx(self):
        """Samples an index from the space
        :return: tuple
        """
        ids = tuple(map(np.random.choice, self.index_shape))
        if len(ids) == 1:
            return ids[0]
        else:
            return ids

    def sample(self):
        """Samples an element from the space
        :return: np.ndarray
        """
        k = self.sample_idx()
        return self[k]

    def __iter__(self):
        """
        :return: DiscretizableSpaceIterator
        """
        return DiscretizableSpaceIterator(self)


class DiscretizableSpaceIterator:
    """
    An iterator over a DiscretizableSpace
    """
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
        """
        Next item
        :return: tuple<(tuple, np.ndarray)>. The first item is the index of the element, and the second is the
            element itself
        """
        index = next(self.index_iter)
        data = self.space[index]
        return (index, data)


class ProductSpace(DiscretizableSpace):
    """
    Handles product of spaces. This class mainly implements the __getitem__ method, and provides some helper functions.
    """
    def __init__(self, *sets):
        """
        Initializer
        :param sets: The list of the sets to take the product. The order matters. Sets can themselves be ProductSpaces.
            Then, the dimensions are flattened.
        """
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
            # We do not use boolean masks for _index_slices because indexing with a mask is only supported for np arrays
            self._index_slices[ns] = slice(current_index, end_index)
            current_index = end_index

    def __getitem__(self, index):
        """
        The indexing method. Indexes are tuples, and each element should be one of the following:
            * an integer. Then, the corresponding dimension is simply indexed by the integer, as would a np.ndarray be,
            * a slice. Then, the corresponding dimension is simply indexed by the slice, as would a np.ndarray be,
            * a np.ndarray of shape (1,). Then, the corresponding dimension has the value in the np.ndarray.
        Finally, if some dimensions are not specified, the index is completed by concatenating as many
        `slice(None, None, None)` to the right as necessary.
        Example: with space = [0,1] x [0,1] x [0,1], with shape (11,?,3)
        space[1,np.ndarray([0.1337])] -> np.ndarray([[0.1, 0.1337, 0],
                                                     [0.1, 0.1337, 0.5],
                                                     [0.1, 0.1337, 1]
                                                    ])
        Note: the typing of the index is only enforced by subclasses. This method does not care what the items in the
        index are.
        :param index: tuple: the index
        :return: np.ndarray
        """
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
            """
            Queries the set corresponding to dimension ns with its corresponding index. In general, the set is
            1-dimensional (Segment or Discrete), since it is a flattened set. Then, the output is of shape (n,1), where
            n is the number of values required by the index
            :param ns: the number of the dimension
            :return: np.ndarray: the elements corresponding to the index on that dimension
            """
            return np.atleast_1d(self._flattened_sets[ns][index[ns]])

        def isnotslice(x):
            """
            Checks whether the argument is a slice
            :param x:
            :return: False iff x is a slice
            """
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
        """
        Returns the component of element x on dimension ns, where ns indexes over the non-flattened sets.
        :param x: np.ndarray
        :param ns: int
        :return:
        """
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
        """
        Returns the component of element x on space target.
        :param x: np.ndarray
        :param target: Space
        :return: np.ndarray: the components of x on target
        """
        if target not in self.sets:
            raise error.InvalidTarget
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
        out = []
        for k in range(len(x_sets)):
            if isinstance(x_sets[k], np.ndarray):
                out += list(x_sets[k].reshape((-1,1)))
            elif isinstance(x_sets[k], tuple):
                out += x_sets[k]
            else:
                out += (x_sets[k], )
        return tuple(out)
