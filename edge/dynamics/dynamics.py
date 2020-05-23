import numpy as np

from .event import EventBased
from edge import error


class DiscreteTimeDynamics(EventBased):
    def __init__(self, stateaction_space):
        self.stateaction_space = stateaction_space

    @property
    def parameters(self):
        raise NotImplementedError

    def step(self, state, action):
        raise NotImplementedError

    def is_feasible_state(self, state):
        raise NotImplementedError

    def compute_map(self):
        Q_map = np.zeros(self.stateaction_space.shape, dtype=int)
        for sa_index, stateaction in iter(self.stateaction_space):
            state, action = self.stateaction_space.get_tuple(stateaction)
            next_state, failed = self.step(state, action)
            next_state_index = self.stateaction_space.state_space.get_index_of(
                next_state
            )
            Q_map[sa_index] = next_state_index
        return Q_map



class TimestepIntegratedDynamics(DiscreteTimeDynamics):
    def __init__(self, stateaction_space):
        super(TimestepIntegratedDynamics, self).__init__(stateaction_space)

    def get_trajectory(self, state, action):
        raise NotImplementedError

    def ensure_in_state_space(self, new_state):
        if new_state in self.stateaction_space.state_space:
            return new_state
        else:
            return self.stateaction_space.state_space.closest_in(new_state)

    def step(self, state, action):
        if (state not in self.stateaction_space.state_space) or (
                action not in self.stateaction_space.action_space):
            raise error.OutOfSpace
        if not self.is_feasible_state(state):
            return state, False

        trajectory = self.get_trajectory(state, action)
        new_state = np.atleast_1d(trajectory.y[:, -1])
        new_state = self.ensure_in_state_space(new_state)
        is_feasible = self.is_feasible_state(new_state)

        return new_state, is_feasible
