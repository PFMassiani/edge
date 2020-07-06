import numpy as np

class Policy:
    def __init__(self, stateaction_space):
        self.stateaction_space = stateaction_space

    def __call__(self, *args, **kwargs):
        return self.get_action(*args, **kwargs)

    def get_action(self, state):
        raise NotImplementedError

    def get_policy_map(self):
        raise NotImplementedError


class RandomPolicy(Policy):
    def __init__(self, stateaction_space):
        super(RandomPolicy, self).__init__(stateaction_space)

    def get_action(self, state):
        action_space = self.stateaction_space.action_space
        def prod(t):
            p = 1
            for e in t:
                p *= e
            return p
        n_available = prod(action_space.shape)
        chosen_action = np.random.choice(n_available)
        return np.unravel_index(chosen_action, action_space.shape)
