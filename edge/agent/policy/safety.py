import numpy as np

from .policy import Policy


class SafetyMaximization(Policy):
    def __init__(self, stateaction_space):
        super(SafetyMaximization, self).__init__(stateaction_space)

    def get_action(self, cautious_probability):
        action_index = np.unravel_index(
            np.argmax(cautious_probability),
            shape=self.stateaction_space.action_space.shape
        )
        action = self.stateaction_space.action_space[action_index]
        return action

    def get_policy_map(self):
        raise NotImplementedError


class SafetyActiveSampling(Policy):
    def __init__(self, stateaction_space):
        super(SafetyActiveSampling, self).__init__(stateaction_space)

    def get_action(self, safety_covariance, is_cautious):
        if not is_cautious.any():
            return None

        # We add some noise so if the covariance is uniform, the sampled
        # action is random
        safety_covariance = safety_covariance + np.random.randn(
            safety_covariance.shape
        ) * 0.01
        cautious_indexes = np.argwhere(is_cautious)
        most_variance_action = np.argmax(
            safety_covariance[cautious_indexes]
        )
        action_idx = tuple(cautious_indexes[most_variance_action])
        action = self.stateaction_space.action_space[action_idx]
        return action

    def get_policy_map(self):
        raise NotImplementedError