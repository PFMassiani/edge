from numpy import atleast_1d

from .environments import Environment
from edge.dynamics import HovershipDynamics
from edge.space import Segment
from edge.reward import ConstantReward


class Hovership(Environment):
    def __init__(self, random_start=False, default_initial_state=None,
                 dynamics_parameters=None, reward=None):
        if dynamics_parameters is None:
            dynamics_parameters = {}
        default_dynamics_parameters = {
            'ground_gravity': 1,
            'gravity_gradient': 0.1,
            'control_frequency': 1,
            'max_thrust': 1,
            'max_altitude': 1,
            'shape': (200, 150)
        }
        default_dynamics_parameters.update(dynamics_parameters)
        dynamics = HovershipDynamics(**default_dynamics_parameters)

        if reward is None:
            # TODO change this so it uses subspaces
            max_altitude = default_dynamics_parameters['max_altitude']
            rewarded_set = Segment(0.8 * max_altitude, max_altitude, 100)
            reward = ConstantReward(
                dynamics.stateaction_space,
                constant=1,
                rewarded_set=rewarded_set
            )

        if default_initial_state is None:
            # TODO change this so it uses stateaction wrappers
            max_altitude = default_dynamics_parameters['max_altitude']
            default_initial_state = atleast_1d(0.75 * max_altitude)

        super(Hovership, self).__init__(
            dynamics=dynamics,
            reward=reward,
            default_initial_state=default_initial_state,
            random_start=random_start
        )
