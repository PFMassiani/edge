import numpy as np
from numbers import Number

from .space import DiscreteSpace, DiscreteProductSpace
from edge import error
from edge.utils import ensure_np


class Segment(DiscreteSpace):
    def __init__(self, low, high, n_points):
        super(Segment, self).__init__()
        if low >= high:
            raise ValueError(f'Bounds {low} and {high} create empty Segment')
        self.low = low
        self.high = high
        self.n_points = n_points
        self.tolerance = (high - low) * 1e-7

    def _get_closest_index(self, x):
        return np.around(
            self.n_points * (x - self.low) / (self.high - self.low)
        )

    def _get_value_of_index(self, index):
        t = index / self.n_points
        return (1 - t) * self.low + t * self.high

    def contains(self, x):
        return (self.low <= x) and (self.high >= x)

    def __getitem__(self, index):
        return self._get_value_of_index(index)

    def indexof(self, x):
        if (self.low > x) or (self.high < x):
            raise error.OutOfSpace
        index = self._get_closest_index(x)
        if abs(self._get_value_of_index(index) - x) > self.tolerance:
            raise error.OutOfSpace(f'Value {x} is between the bounds of the '
                                   'Segment, but is not on the '
                                   'discretization grid')

        return index

    def get_index_iterator(self):
        return iter(range(self.n_points + 1))

    def sample(self):
        return np.random.choice(self.n_points + 1)


class Box(DiscreteProductSpace):
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
