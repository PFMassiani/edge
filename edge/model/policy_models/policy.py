import numpy as np
from edge.error import OutOfSpace


class Policy:
    """
    Base class for all Policies
    A Policy is a callable object that gives the next action that should be taken.
    """
    def __init__(self, stateaction_space):
        self.stateaction_space = stateaction_space

    def __call__(self, *args, **kwargs):
        """
        Returns the next action. The parameters should be the same as self.get_action.
        :return: the next action
        """
        return self.get_action(*args, **kwargs)

    def get_action(self, state):
        """ Abstract method
        Returns the next action. Subclasses may require more parameters. If the Policy cannot find a suitable action,
        it should return None.
        :param state: the current state
        :return: None or next action
        """
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
        return action_space[np.unravel_index(chosen_action, action_space.shape)]


class ConstantPolicy(Policy):
    def __init__(self, stateaction_space, constant_action):
        super().__init__(stateaction_space)
        if constant_action not in stateaction_space.action_space:
            raise OutOfSpace(f'The constant action {constant_action} is not '
                             f'valid')
        self.constant_action = constant_action

    def get_action(self, state):
        return self.constant_action


class AffinePolicy(Policy):
    def __init__(self, stateaction_space, offset, jacobian):
        super().__init__(stateaction_space)
        self.s0, self.a0 = offset
        self.jacobian = jacobian

    def get_action(self, state):
        a = self.a0 + self.jacobian @ (state - self.s0)
        a = self.stateaction_space.action_space.closest_in(a)
        return a