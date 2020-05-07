from .event import EventBased
from edge import error


class DiscreteTimeDynamics(EventBased):
    def __init__(self, stateaction_space):
        self.stateaction_space = stateaction_space

    def step(self, state, action):
        raise NotImplementedError

    def is_feasible_state(self, state):
        raise NotImplementedError


class TimestepIntegratedDynamics(DiscreteTimeDynamics):
    def __init__(self, stateaction_space):
        super(TimestepIntegratedDynamics, self).__init__(stateaction_space)

    def get_trajectory(self, state, action):
        raise NotImplementedError

    def ensure_on_feasible_set(self, new_state):
        raise NotImplementedError

    def step(self, state, action):
        if (state not in self.stateaction_space.state_space) or (
                action not in self.stateaction_space.action_space):
            raise error.OutOfSpace
        if not self.is_feasible_state(state):
            return state, True

        trajectory = self.get_trajectory(state, action)
        new_state = trajectory.y[:, -1]
        new_state = self.ensure_on_feasible_set(new_state)

        return new_state
