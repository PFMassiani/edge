import unittest
import numpy as np

from edge.dynamics import SlipDynamics
from edge.envs import Slip


class TestSlip(unittest.TestCase):
    def test_default_creation(self):
        slip = Slip()
        self.routine(slip, slip.default_initial_state)


    def routine(self, slip, initial_state):
        self.assertTrue(not slip.in_failure_state)
        self.assertTrue(slip.feasible)
        self.assertTrue(not slip.has_failed)
        self.assertEqual(slip.default_initial_state[0], initial_state[0])
        self.assertEqual(slip.s, slip.default_initial_state)

        # * jump up and down 100 times
        total_reward = 0
        MAX_ITER = 100
        action = np.array([0.0])
        for i in range(MAX_ITER):
            s, r, failed = slip.step(slip.action_space[action])
            total_reward += r
            print(s)

        self.assertEqual(total_reward, 1.0*MAX_ITER)
        self.assertTrue(np.isclose(s, initial_state)[0])