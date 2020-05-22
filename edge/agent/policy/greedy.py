import numpy as np

from .policy import Policy


class ConstrainedEpsilonGreedy(Policy):
    def __init__(self, env, greed):
        super(ConstrainedEpsilonGreedy, self).__init__(env)
        self.__greed = greed

    @property
    def greed(self):
        return self.__greed

    @greed.setter
    def greed(self, new_greed):
        self.__greed = np.clip(new_greed, 0, 1)

    def get_action(self, q_values, constraint):
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
            )

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
                self.env.action_space.shape
            )
            action = self.env.action_space[action_index]
            return action
        else:
            return None

    def get_policy_map(self, q_values, constraint):
        raise NotImplementedError


class EpsilonGreedy(ConstrainedEpsilonGreedy):
    def __init__(self, env, greed):
        super(EpsilonGreedy, self).__init__(env, greed)

    def get_action(self, q_values):
        constraint = np.ones(self.env.action_space.shape, dtype=bool)
        return super(EpsilonGreedy, self).get_action(q_values, constraint)
