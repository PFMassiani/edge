import numpy as np
from scipy.integrate import solve_ivp

from edge.space import Box, Discrete, StateActionSpace
from .dynamics import DiscreteTimeDynamics, TimestepIntegratedDynamics
from .event import event
from edge import error


class HovershipDynamics(TimestepIntegratedDynamics):
    def __init__(self, ground_gravity, gravity_gradient, control_frequency,
                 max_thrust, max_altitude, shape=(200, 150)):
        stateaction_space = StateActionSpace.from_product(
            Box([0, 0], [max_altitude, max_thrust], shape)
        )
        super(HovershipDynamics, self).__init__(stateaction_space)
        self.ground_gravity = ground_gravity
        self.gravity_gradient = gravity_gradient
        self.control_frequency = control_frequency
        self.ceiling_value = stateaction_space.state_space.high
        self.max_thrust = stateaction_space.action_space.high

    def is_feasible_state(self, state):
        if state not in self.stateaction_space.state_space:
            raise error.OutOfSpace
        return True

    @property
    def parameters(self):
        return {
            'ground_gravity': self.ground_gravity,
            'gravity_gradient': self.gravity_gradient,
            'control_frequency': self.control_frequency,
            'max_thrust': self.max_thrust,
            'max_altitude': self.ceiling_value,
            'shape': self.stateaction_space.shape
        }

    @event(True, 1)
    def ceiling(self, t, y):
        return y - self.ceiling_value

    def get_force_on_ship(self, state, action):
        grav_field = np.max([
            0,
            np.tanh(0.75 * (self.ceiling_value - state))
        ]) * self.gravity_gradient
        total_force = - self.ground_gravity - grav_field + action
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


class DiscreteHovershipDynamics(DiscreteTimeDynamics):
    def __init__(self, ground_gravity, gravity_gradient, max_thrust,
                 max_altitude, minimum_gravity_altitude,
                 maximum_gravity_altitude):
        stateaction_space = StateActionSpace(
            Discrete(max_altitude + 1),
            Discrete(max_thrust + 1)
        )
        super(DiscreteHovershipDynamics, self).__init__(stateaction_space)
        self.ground_gravity = ground_gravity
        self.gravity_gradient = gravity_gradient
        self.minimum_gravity_altitude = minimum_gravity_altitude
        self.maximum_gravity_altitude = maximum_gravity_altitude

    def is_feasible_state(self, state):
        if state not in self.stateaction_space.state_space:
            raise error.OutOfSpace
        return True

    def step(self, state, action):
        if (state not in self.stateaction_space.state_space) or (
                action not in self.stateaction_space.action_space):
            raise error.OutOfSpace
        if not self.is_feasible_state(state):
            return state, False
        z = state[0]
        if z >= self.minimum_gravity_altitude:
            gravity_field = np.zeros_like(state)
        elif z <= self.maximum_gravity_altitude:
            gravity_field = np.ones_like(state) * self.gravity_gradient * (
                self.minimum_gravity_altitude - self.maximum_gravity_altitude
            )
        else:
            gravity_field = np.ones_like(state) * self.gravity_gradient * (
                self.minimum_gravity_altitude - z
            )

        dynamics_step = action - self.ground_gravity - gravity_field

        new_state = self.stateaction_space.state_space.closest_in(
            state + dynamics_step
        )
        is_feasible = self.is_feasible_state(new_state)
        return new_state, is_feasible
