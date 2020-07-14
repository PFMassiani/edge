import numpy as np

from .policy import Policy


class SafetyMaximization(Policy):
    """
    Policy that maximizes safety: it picks the action with highest probability of being safe.
    """
    def __init__(self, stateaction_space):
        super(SafetyMaximization, self).__init__(stateaction_space)

    def get_action(self, cautious_probability):
        """
        Returns the next action
        :param cautious_probability: np.ndarray of shape action_space.shape. The probability that the action is safe,
            for any action
        :return: next_action
        """
        action_index = np.unravel_index(
            np.argmax(cautious_probability),
            shape=self.stateaction_space.action_space.shape
        )
        action = self.stateaction_space.action_space[action_index]
        return action

    def get_policy_map(self):
        raise NotImplementedError


class SafetyActiveSampling(Policy):
    """
    Policy that maximizes the information gain about the safety measure: it picks the action with highest variance.
    """
    def __init__(self, stateaction_space):
        super(SafetyActiveSampling, self).__init__(stateaction_space)

    def get_action(self, safety_covariance, is_cautious):
        """
        Returns the next action, or None if no action was found
        :param safety_covariance: the covariance of the actions in the starting state
        :param is_cautious: boolean array of whether an action is cautious, i.e., estimated safe
        :return: None or next_action
        """
        # If no action is safe, this policy does not know what to do
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