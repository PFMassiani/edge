import numpy as np
from scipy.integrate import solve_ivp

from edge.space import Box, Discrete, StateActionSpace
from .dynamics import DiscreteTimeDynamics, TimestepIntegratedDynamics
from .event import event
from edge import error


class SlipDynamics(DiscreteTimeDynamics):
    def __init__(self,
                 gravity,
                 mass,
                 stiffness,
                 resting_length,
                 energy,
                 failed=False,
                 state_bounds=(0.0, 1),
                 action_bounds=(-1/18*np.pi, 7/18*np.pi),
                 shape=(200, 100)):
        stateaction_space = StateActionSpace.from_product(
            Box([state_bounds[0], action_bounds[0]],
                [state_bounds[1], action_bounds[1]], shape)
        )
        super(SlipDynamics, self).__init__(stateaction_space)
        self.gravity = gravity
        self.mass = mass
        self.stiffness = stiffness
        self.resting_length = resting_length
        self.energy = energy
        self.failed = False

    def is_feasible_state(self, state):
        if state not in self.stateaction_space.state_space:
            raise error.OutOfSpace
        return True

    @property
    def parameters(self):
        return {
            'gravity': self.gravity,
            'mass': self.mass,
            'stiffness': self.stiffness,
            'resting_length': self.resting_length,
            'energy': self.energy,
            'shape': self.stateaction_space.shape
        }

    def step(self, state, action):
        # low-dimensional representation:
        # state is normalized height, state = m*g*h = m*g*x[1]
        state = np.atleast_1d(state)
        action = np.atleast_1d(action)
        # * set some simulation parameters
        MAX_TIME = 3

        # * map to high-dimensional state
        # forward velocity
        y = state[0]*self.energy/self.mass/self.gravity
        v = np.sqrt(2/self.mass*(self.energy - self.energy*state[0]))
        x0 = np.array([0, y,  # body position
                      v, 0,  # body velocity
                      0, 0])  # foot position (fill in below)
        x0[4] = x0[0] + np.sin(action)*self.resting_length
        x0[5] = x0[1] - np.cos(action)*self.resting_length

        # * define high-dimensional dynamics

        # Helper functions for traj dynamics
        def fall_event(t, y):
            return y[1]
        fall_event.terminal = True
        fall_event.direction = -1

        def touchdown_event(t, y):
            return y[5]
        touchdown_event.terminal = True
        touchdown_event.direction = -1

        def liftoff_event(t, y):
            return np.hypot(y[0]-y[4], y[1]-y[5]) - self.resting_length
        liftoff_event.terminal = True
        liftoff_event.direction = 1

        def apex_event(t, y):
            return y[3]
        apex_event.terminal = True
        apex_event.direction = -1

        def flight(t, y):
            return np.array([y[2], y[3], 0, -self.gravity, y[2], y[3]])

        def stance(t, y):
            alpha = np.arctan2(y[1] - y[5], y[0] - y[4]) - np.pi/2.0
            spring_length = np.hypot(y[0]-y[4], y[1]-y[5])
            leg_force = self.stiffness/self.mass*(self.resting_length
                                                  - spring_length)
            xdotdot = -leg_force*np.sin(alpha)
            ydotdot = leg_force*np.cos(alpha) - self.gravity
            return np.array([y[2], y[3], xdotdot, ydotdot, 0, 0])

        # TODO: implement jacobian of stance

        # * simulate:

        while True:
            # while statement is just to allow breaks. It does not loop

            # * FLIGHT: simulate till touchdown
            events = [fall_event, touchdown_event]
            traj = solve_ivp(flight, t_span=[0, 0+MAX_TIME],
                             y0=x0, events=events, max_step=0.01)

            # if you fell, stop now
            if traj.t_events[0].size != 0:  # if empty
                break

            # * STANCE: simulate till liftoff
            events = [fall_event, liftoff_event]
            traj = solve_ivp(stance, t_span=[traj.t[-1], traj.t[-1]+MAX_TIME],
                             y0=traj.y[:, -1], events=events, max_step=0.0005)

            # if you fell, stop now
            if traj.t_events[0].size != 0:  # if empty
                break

            # * FLIGHT: simulate till apex
            events = [fall_event, apex_event]
            traj = solve_ivp(flight, t_span=[traj.t[-1], traj.t[-1]+MAX_TIME],
                             y0=traj.y[:, -1], events=events, max_step=0.01)

            break

        # * Check if low-level failured conditions are triggered
        # point mass touches the ground, or reverses direction
        if traj.y[1, -1] < 1e-3 or traj.y[2, -1] < 0:
            self.failed = True

        # * map back to high-level state.
        new_state = self.mass*self.gravity*traj.y[1, -1]/self.energy
        new_state = np.atleast_1d(new_state)
        new_state = self.stateaction_space.state_space.closest_in(new_state)

        return new_state, self.is_feasible_state(new_state)
