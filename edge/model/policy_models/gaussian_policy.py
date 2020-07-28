import numpy as np

from . import Policy
from ..inference import GaussianDensity


class GaussianPolicy(Policy):
    def __init__(self, stateaction_space, discount_rate, step_size,
                 features_function=None, n_features=None, initial_weight=0, initial_var=1):
        super(GaussianPolicy, self).__init__(stateaction_space)
        self.actions_density = GaussianDensity(
            stateaction_space.action_space.index_dim, features_function, n_features, initial_weight, initial_var
        )

        self.discount_rate = discount_rate
        self.step_size = step_size

    def get_action(self, state):
        action = self.actions_density(state)
        # This projection step is here because our action space is bounded
        # The effect of this is assigning the 'overflowing' probability mass to the action on the boundary
        # This may lead to taking such actions too often for distributions with high variance
        action = self.stateaction_space.action_space.closest_in(action)
        return action

    def update(self, episode):
        T = len(episode)
        rewards = np.array([episode[t]['reward'] for t in range(T)])
        discounts = np.array([self.discount_rate**t for t in range(T)])
        for t in range(T):
            state = episode[t]['state']
            action = episode[t]['action']
            g = discounts[:T-t-1] @ rewards[:T-t-1]
            grad_of_log_function = self.actions_density.gradient_of_log(state)
            gradient_step = self.step_size * discounts[t] * g * grad_of_log_function(action)
            self.actions_density.update(gradient_step)

    def __str__(self):
        return f'GaussianPolicy(discount_rate={self.discount_rate}, step_size={self.step_size})'

    def __repr__(self):
        return str(self)