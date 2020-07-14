import numpy as np

from .reward import Reward


class AffineReward(Reward):
    """
    Defines an affine reward on the StateActionSpace
    """
    def __init__(self, stateaction_space, ranges_along_dim):
        """
        Initializer
        Example:
        ranges_along_dim = [(0,1),(2.71, 3.14)] defines an affine reward ranging between 0 and 1 on the first dimension,
            and 2.71 and 3.14 on the second one.
        :param stateaction_space: the stateaction_space
        :param ranges_along_dim: list of 2-dimensional tuples giving the range along each dimension. Missing dimensions
            are completed with (0,0).
        """
        super(AffineReward, self).__init__(stateaction_space)
        n_missing_dims = self.stateaction_space.index_dim - len(
            ranges_along_dim
        )
        missing_ranges = [(0, 0)] * n_missing_dims
        ranges_along_dim = np.array(ranges_along_dim + missing_ranges)
        self.origins_rewards = ranges_along_dim[:, 0]
        self.ends_rewards = ranges_along_dim[:, 1]

        limits = np.array(self.stateaction_space.limits)
        self.origins = limits[:, 0]
        self.ends = limits[:, 1]
        self.extents = self.ends - self.origins

    def get_reward(self, state, action, new_state, failed):
        stateaction = self.stateaction_space[state, action]
        reward_vector = (
                (stateaction - self.origins) / self.extents
        ) * (self.ends_rewards - self.origins_rewards) + self.origins_rewards
        return reward_vector.sum()