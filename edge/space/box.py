import numpy as np
from numbers import Number

from .space import DiscretizableSpace, ProductSpace
from edge import error
from edge.utils import ensure_np


class Segment(DiscretizableSpace):
    def __init__(self, low, high, n_points):
        super(Segment, self).__init__(
            index_dim=1,
            discretization_shape=n_points
        )
        if low >= high:
            raise ValueError(f'Bounds {low} and {high} create empty Segment')
        self.low = low
        self.high = high
        self.n_points = n_points
        self.tolerance = (high - low) * 1e-7

    def _get_closest_index(self, x):
        return int(np.around(
            (self.n_points - 1) * (x - self.low) / (self.high - self.low)
        ))

    def _get_value_of_index(self, index):
        t = index / (self.n_points - 1)
        return (1 - t) * self.low + t * self.high

    def contains(self, x):
        return (self.low <= x) and (self.high >= x)

    def is_on_grid(self, x):
        if x not in self:
            return False

        closest_index = self._get_closest_index(x)
        if abs(self[closest_index] - x) > self.tolerance:
            return False

        return True

    def __getitem__(self, index):
        if (index < 0) or (index >= self.n_points):
            raise IndexError('Space index out of range')
        return np.atleast_1d(self._get_value_of_index(index))

    def get_index_of(self, x, around_ok=False):
        if x not in self:
            raise error.OutOfSpace
        index = self._get_closest_index(x)
        if around_ok:
            return index
        elif self.is_on_grid(x):
            return index
        else:
            raise error.NotOnGrid

    def get_index_iterator(self):
        return iter(range(self.n_points))

    def sample_idx(self):
        return np.random.choice(self.n_points)

    def closest_in(self, x):
        if x in self:
            return x
        elif x < self.low:
            return self.low
        elif x > self.high:
            return self.high


class Box(ProductSpace):
    def __init__(self, low, high, shape):
        if isinstance(low, Number) and isinstance(high, Number):
            self.dim = len(shape)
            low = np.array([low] * self.dim)
            high = np.array([high] * self.dim)
        else:
            low = ensure_np(low)
            high = ensure_np(high)
            if not (low.shape == high.shape) and (low.shape == shape):
                raise ValueError(f'Shape mismatch. Low {low.shape} High '
                                 '{high.shape} Shape {shape}')
            self.dim = len(shape)

        self.segments = [None] * self.dim
        for d in range(self.dim):
            self.segments[d] = Segment(low[d], high[d], shape[d])

        super(Box, self).__init__(*self.segments)
