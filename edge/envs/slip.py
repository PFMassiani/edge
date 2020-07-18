from numpy import atleast_1d

from .environments import Environment
from edge.dynamics import SlipDynamics
from edge.reward import ConstantReward, AffineReward


class Slip(Environment):
    def __init__(self, random_start=False, default_initial_state=None,
                 dynamics_parameters=None):
        if dynamics_parameters is None:
            dynamics_parameters = {}

        default_dynamics_parameters = {
            'gravity': 9.81,
            'mass': 80.0,
            'stiffness': 8200.0,
            'resting_length': 1.0,
            'energy': 1877.08,
            'shape': (200, 10)
        }
        default_dynamics_parameters.update(dynamics_parameters)
        dynamics = SlipDynamics(**default_dynamics_parameters)

        # just give it a reward every step, no matter what
        # this is equivalent to "hop as long as possible"
        reward = ConstantReward(dynamics.stateaction_space, constant=1)
        # now let's incentivize going as fast as possible.
        reward += AffineReward(dynamics.stateaction_space, [[1, 0], [0, 0]])

        if default_initial_state is None:
            # TODO change this so it uses stateaction wrappers
            # not sure what the above is referring to...
            # max_altitude = default_dynamics_parameters['max_altitude']
            # standing_energy = (default_dynamics_parameters['resting_length'] *
            #                    default_dynamics_parameters['mass'] *
            #                    default_dynamics_parameters['gravity'] /
            #                    default_dynamics_parameters['energy'])

            default_initial_state = atleast_1d(0.4)

        super(Slip, self).__init__(
            dynamics=dynamics,
            reward=reward,
            default_initial_state=default_initial_state,
            random_start=random_start
        )

    def is_failure_state(self, state):
        # TODO change so it uses stateaction wrappers

        return state <= 1e-3
