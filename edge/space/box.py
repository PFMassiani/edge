import numpy as np
from numbers import Number

from .space import DiscretizableSpace, ProductSpace
from edge import error


class Segment(DiscretizableSpace):
    """The simplest Space. It models a segment.
    Note that unbounded intervals are not supported.
    """
    def __init__(self, low, high, n_points):
        """
        :param low: the lowest value of the segment
        :param high: the highest value of the segment
        :param n_points: the number of points to consider in the discretization
        """
        if low >= high:
            raise ValueError(f'Bounds {low} and {high} create empty Segment')
        super(Segment, self).__init__(index_shape=(n_points,))
        self.low = low
        self.high = high
        self.n_points = n_points
        self.tolerance = (high - low) * 1e-7  # For approximate check of whether a point is on the grid

    def __getitem__(self, index):
        """
        :param index: 1-element tuple, np.ndarray, or slice. If np.ndarray, this method just feeds the value through
        :return:
        """
        if isinstance(index, tuple):
            if len(index) == 1:
                index = index[0]
            else:
                # This brings us to the else clause of the next if : nothing needs to be done
                pass

        if isinstance(index, np.ndarray):
            elem = index
            if elem in self:
                return elem
            else:
                raise IndexError(f'Index {index} is understood as an element '
                                 'of the Space and does not belong to it')
        elif isinstance(index, (int, np.integer)):
            # We allow negative indexing, as when indexing lists or np.ndarrays
            if index < 0:
                index += self.n_points
            if index < 0 or index > self.n_points - 1:
                raise IndexError(f"Index {index} is out of bounds for Segment "
                                 f"with length {self.n_points}")
            else:
                return np.atleast_1d(self._get_value_of_index(index))
        elif isinstance(index, slice):
            rangeargs = index.indices(self.n_points)
            return np.array([
                np.atleast_1d(self._get_value_of_index(i))
                for i in range(*rangeargs)
            ])
        else:
            raise TypeError('Index can only be numpy ndarray, int, slice, '
                            f'or 1d tuple, not {type(index)}')

    def _get_closest_index(self, x):
        return int(np.around(
            (self.n_points - 1) * (x - self.low) / (self.high - self.low)
        ))

    def _get_value_of_index(self, index):
        t = index / (self.n_points - 1)
        return (1 - t) * self.low + t * self.high

    def contains(self, x):
        if x.shape != (1,):
            return False
        else:
            return (self.low <= x[0]) and (self.high >= x[0])

    def is_on_grid(self, x):
        if x not in self:
            return False
        closest_index = self._get_closest_index(x)
        return np.all(np.abs(self[closest_index] - x) <= self.tolerance)

    def get_index_of(self, x, around_ok=False):
        if x not in self:
            raise error.OutOfSpace
        index = self._get_closest_index(x)
        if around_ok:
            return index
        elif np.all(self.is_on_grid(x)):
            return index
        else:
            raise error.NotOnGrid

    def closest_in(self, x):
        return np.clip(x, self.low, self.high)

    @property
    def limits(self):
        return self.low, self.high


class Box(ProductSpace):
    """A product of segments"""
    def __init__(self, low, high, shape):
        """
        Initializer
        If low (resp. high) is a float, then all Segments have a lower (resp. higher) bound equal to that float.
        Otherwise, low (resp. high) should be a tuple whose values are the lower (resp. higher) bound on the current
        dimension.
        If low and high are tuples, then their length should be equal to the length of the shape argument.
        Note: the case when only one of low and high is a tuple is not supported. They need to be in the same format.
        :param low: float or tuple
        :param high: float or tuple
        :param shape: The discretization shape.
        """
        if isinstance(low, Number) and isinstance(high, Number):
            self.dim = len(shape)
            low = np.array([low] * self.dim)
            high = np.array([high] * self.dim)
        else:
            low = np.array(low)
            high = np.array(high)
            if not (low.shape == high.shape) and (low.shape == shape):
                raise ValueError(f'Shape mismatch. Low {low.shape} High '
                                 '{high.shape} Shape {shape}')
            self.dim = len(shape)

        self.segments = [None] * self.dim
        for d in range(self.dim):
            self.segments[d] = Segment(low[d], high[d], shape[d])

        super(Box, self).__init__(*self.segments)
