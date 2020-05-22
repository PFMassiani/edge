import numpy as np

from .reward import Reward


class AffineReward(Reward):
    def __init__(self, stateaction_space, ranges_along_dim):
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