import numpy as np

from . import Policy


class ConstrainedEpsilonGreedy(Policy):
    """
    Defines a policy that is epsilon-greedy, but with a hard constraint on the state-action space: in a given state,
    only a subset of the actions may be available.
    The parameter `greed` is the eps: 1 for only exploration, 0 for only greed.
    """
    def __init__(self, stateaction_space, greed):
        super(ConstrainedEpsilonGreedy, self).__init__(stateaction_space)
        self.__greed = greed

    @property
    def greed(self):
        return self.__greed

    @greed.setter
    def greed(self, new_greed):
        self.__greed = np.clip(new_greed, 0, 1)

    def get_action(self, q_values, constraint):
        """
        Returns the next action, or None if no action was found
        :param q_values: np.ndarray: the values of the state-action pairs in the starting state
        :param constraint: np.ndarray: boolean array of whether an action satisfies the constraint
        :return: None or next_action
        """
        # TODO remove this function
        def choose_action(q_values_extract):
            n = len(q_values_extract)
            best_value = np.argmax(q_values_extract)
            probabilities = np.ones(n) * self.greed / n
            probabilities[best_value] += 1 - self.greed
            return np.random.choice(n, p=probabilities)

        n_available = sum(constraint)
        if n_available > 0:
            available_to_all_lookup = np.atleast_1d(
                np.argwhere(constraint).squeeze()
            )  # lookup table to find the indexes of available actions among all the actions

            available_q_values = q_values[constraint]
            best_value = np.argmax(available_q_values)
            probabilities = np.ones(n_available) * self.greed / n_available
            probabilities[best_value] += 1 - self.greed
            action_index_in_available = np.random.choice(
                n_available,
                p=probabilities
            )
            raveled_index = available_to_all_lookup[action_index_in_available]
            action_index = np.unravel_index(
                raveled_index,
                self.stateaction_space.action_space.shape
            )
            action = self.stateaction_space.action_space[action_index]
            return action
        else:
            return None

    def get_policy_map(self, q_values, constraint):
        raise NotImplementedError


class EpsilonGreedy(ConstrainedEpsilonGreedy):
    """
    Defines a policy that is epsilon-greedy.
    The parameter `greed` is the eps: 1 for only exploration, 0 for only greed.
    """
    def __init__(self, stateaction_space, greed):
        super(EpsilonGreedy, self).__init__(stateaction_space, greed)

    def get_action(self, q_values):
        """
        Returns the next action
        :param q_values: np.ndarray: the values of the state-action pairs in the starting state
        :return: next action
        """
        constraint = np.ones(self.stateaction_space.action_space.shape, dtype=bool)
        return super(EpsilonGreedy, self).get_action(q_values, constraint)
