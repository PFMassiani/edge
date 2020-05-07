import numpy as np

from edge.space import Segment
from .dynamics import TimestepIntegratedDynamics
from .event import event
from edge import error


class HovershipDynamics(TimestepIntegratedDynamics):
    def __init__(self, stateaction_space,
                 ground_gravity, gravity_gradient, control_frequency
                 ):
        if not isinstance(stateaction_space.state_space, Segment):
            raise TypeError('This dynamics only supports Segment state spaces')
        super(TimestepIntegratedDynamics, self).__init__(stateaction_space)
        self.ground_gravity = ground_gravity
        self.gravity_gradient = gravity_gradient
        self.control_frequency = control_frequency

    def is_feasible_state(self, state):
        if state not in self.stateaction_space.state_space:
            raise error.OutOfSpace
        return True

    def ensure_on_feasible_set(self, new_state):
        if new_state in self.stateaction_space.state_space:
            return new_state
        else:
            return self.stateaction_space.state_space.closest_in(new_state)

    @event(True, 1)
    def ceiling(self, t, y):
        return y - self.stateaction_space.state_space.high

    def timecontinuous_dynamics(self, t, state, action):
        gravity_correction = - np.tanh(0.75 * self.ceiling - state
                                       ) * self.gravity_gradient
        corrected_gravity = self.ground_gravity + min(0, gravity_correction)
        total_force = - corrected_gravity + action
