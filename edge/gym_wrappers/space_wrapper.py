import numpy as np

from ..space import Box, Discrete


class BoxWrapper(Box):
    DEFAULT_INF_CEILING = 100
    def __init__(self, gym_box, discretization_shape=None, inf_ceiling=None):
        self.inf_ceiling = BoxWrapper.DEFAULT_INF_CEILING if inf_ceiling is None else inf_ceiling
        self.gym_space = gym_box
        self.gym_shape = self.gym_space.shape
        lows = self._clip_inf(self.gym_space.low.reshape(-1))
        highs = self._clip_inf(self.gym_space.high.reshape(-1))
        if discretization_shape is None:
            discretization_shape = tuple([100 for _ in range(lows.shape[0])])
        if lows.shape[0] != len(discretization_shape):
            raise ValueError(f'Dimension mismatch: Gym Box has {lows.shape[0]} dimensions, while the Edge '
                             f'one has {len(discretization_shape)}')
        super(BoxWrapper, self).__init__(lows, highs, discretization_shape)

    def _clip_inf(self, a):
        return np.nan_to_num(a, posinf = self.inf_ceiling, neginf= - self.inf_ceiling)

    def to_gym(self, index):
        return self[index].reshape(self.gym_shape)

    def from_gym(self, gym_element):
        return gym_element.reshape(-1)


class DiscreteWrapper(Discrete):
    def __init__(self, gym_discrete):
        super(DiscreteWrapper, self).__init__(gym_discrete.n, 0, gym_discrete.n - 1)
        self.gym_space = gym_discrete

    def to_gym(self, index):
        return int(self[index])

    def from_gym(self, gym_element):
        return float(gym_element)
