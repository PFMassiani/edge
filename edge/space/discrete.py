import numpy as np

from . import DiscretizableSpace
from edge import error


class Discrete(DiscretizableSpace):
    """
    A collection of linearly spaced points. Similar to np.linspace.
    """
    def __init__(self, n, start=0, end=None):
        if end is None:
            end = n - 1
        if start > end:
            raise ValueError('Ill-ordered start and end values for Discrete')
        super(Discrete, self).__init__(index_shape=(n,))

        if start != 0 or end != n - 1:
            discretization = np.linspace(start, end, n).reshape((-1, 1))
        else:
            discretization = np.arange(n).reshape((-1, 1))

        self.__discretization = discretization
        self.n = n
        self.start = self.__discretization[0,0]
        self.end = self.__discretization[-1,0]

    def __getitem__(self, index):
        if isinstance(index, np.ndarray):
            elem = index
            if elem in self:
                return elem
            else:
                raise IndexError(f'Index {index} is understood as an element '
                                 'of the Space and does not belong to it')

        elif isinstance(index, (int, np.integer, slice, tuple)):
            return self.__discretization[index]

        else:
            raise TypeError('Index can only be numpy ndarray, int or slice, '
                            f'not {type(index)}')

    def contains(self, x):
        if x.shape != (1,):
            return False
        return x[0] in self.__discretization

    def is_on_grid(self, x):
        return x in self

    def get_index_of(self, x, around_ok=False):
        if x not in self:
            raise error.OutOfSpace

        return np.argmax(self.__discretization == x)

    def closest_in(self, x):
        return self[np.argmin(np.abs(self.__discretization - x))]

    @property
    def limits(self):
        return (self.start, self.end)