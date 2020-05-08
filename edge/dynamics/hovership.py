import numpy as np
from scipy.integrate import solve_ivp

from edge.space import Box, StateActionSpace
from .dynamics import TimestepIntegratedDynamics
from .event import event
from edge import error


class HovershipDynamics(TimestepIntegratedDynamics):
    def __init__(self, ground_gravity, gravity_gradient, control_frequency,
                 max_thrust, max_altitude, shape=(200, 150)):
        stateaction_space = StateActionSpace.from_product(
            Box([0, 0], [max_altitude, max_thrust], shape)
        )
        super(TimestepIntegratedDynamics, self).__init__(stateaction_space)
        self.ground_gravity = ground_gravity
        self.gravity_gradient = gravity_gradient
        self.control_frequency = control_frequency
        self.ceiling_value = stateaction_space.state_space.high

    def is_feasible_state(self, state):
        if state not in self.stateaction_space.state_space:
            raise error.OutOfSpace
        return True

    def ensure_in_feasible_set(self, new_state):
        if new_state in self.stateaction_space.state_space:
            return new_state
        else:
            return self.stateaction_space.state_space.closest_in(new_state)

    @event(True, 1)
    def ceiling(self, t, y):
        return y - self.ceiling_value

    def get_force_on_ship(self, state, action):
        gravity_correction = - np.tanh(0.75 * self.ceiling_value - state
                                       ) * self.gravity_gradient
        corrected_gravity = self.ground_gravity + min(0, gravity_correction)
        total_force = - corrected_gravity + action
        return total_force

    def get_trajectory(self, state, action):
        def dynamics(t, y):
            return self.get_force_on_ship(y, action)

        max_integration_time = 1. / self.control_frequency
        trajectory = solve_ivp(
            fun=dynamics,
            t_span=(0, max_integration_time),
            y0=np.atleast_1d(state),
            events=self.get_events()
        )
        return trajectory
