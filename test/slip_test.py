import unittest
from inspect import signature
import numpy as np

from edge.dynamics import SlipDynamics


def number_of_parameters(func):
    sig = signature(func)
    params = sig.parameters
    return len(params)


class SlipTests(unittest.TestCase):

    def test_still_slip(self):
        slip_dynamics = SlipDynamics(
            gravity=9.81,
            mass=80,
            stiffness=8200,
            resting_length=1.,
            energy=1877.0
        )
        # state_space = slip_dynamics.stateaction_space.state_space

        initial_state = slip_dynamics.stateaction_space.state_space.element(1.0)
        # initial_state = np.atleast_1d(1.0)
        action = slip_dynamics.stateaction_space.action_space.element(0.0)
        new_state, feasible = slip_dynamics.step(initial_state, action)

        self.assertTrue(slip_dynamics.is_feasible_state(initial_state))
        self.assertTrue(slip_dynamics.is_feasible_state(new_state))
        self.assertTrue(np.isclose(initial_state-new_state, 0.0)[0])
