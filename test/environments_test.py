from numpy import atleast_1d

import unittest

from edge.envs import Hovership, DiscreteHovership


class TestHovership(unittest.TestCase):
    def test_default_creation(self):
        hovership = Hovership()
        self.routine(hovership, hovership.default_initial_state)

    def test_custom_creation(self):
        dynamics_parameters = {
            'ground_gravity': 0,
            'gravity_gradient': 0,
        }
        default_initial_state = atleast_1d(0.1)
        hovership = Hovership(
            random_start=True,
            default_initial_state=default_initial_state,
            dynamics_parameters=dynamics_parameters
        )
        hovership.reset(s=default_initial_state)
        self.routine(hovership, default_initial_state)

        for t in range(10):
            hovership.reset()
            if hovership.s[0] != default_initial_state[0]:
                break
        else:
            self.assertTrue(False)

        hovership.reset(s=default_initial_state)
        hovership.step(action=atleast_1d(0.))
        self.assertEqual(hovership.s[0], default_initial_state[0])

    def test_stateaction_space_stability(self):
        hovership = Hovership()
        for t in range(10):
            s, r, failed = hovership.step(atleast_1d(1.))
            self.assertTrue(s in hovership.stateaction_space.state_space)

        hovership.reset()
        for t in range(100):
            s, r, failed = hovership.step(atleast_1d(0.))
            self.assertTrue(s in hovership.stateaction_space.state_space)

    def routine(self, hovership, initial_state):
        self.assertTrue(not hovership.in_failure_state)
        self.assertTrue(hovership.feasible)
        self.assertTrue(not hovership.has_failed)
        self.assertEqual(hovership.default_initial_state[0], initial_state[0])
        self.assertEqual(hovership.s, hovership.default_initial_state)

        r = 0
        MAX_ITER = 1e2
        t = 0
        while r == 0 and t < MAX_ITER:
            s, r, failed = hovership.step(atleast_1d(1))
            t += 1
        self.assertTrue(hovership.s[0] > hovership.default_initial_state[0])
        self.assertTrue(not failed)
        self.assertEqual(r, 1)

        hovership.reset(atleast_1d(0))
        self.assertEqual(hovership.s, atleast_1d(0.))
        self.assertTrue(hovership.has_failed)
        self.assertTrue(hovership.in_failure_state)

        hovership.step(atleast_1d(1))
        self.assertEqual(hovership.s, atleast_1d(0.))
        self.assertTrue(hovership.has_failed)
        self.assertTrue(hovership.in_failure_state)


class TestDiscreteHovership(unittest.TestCase):
    def test_default_creation(self):
        hovership = DiscreteHovership()
        self.routine(hovership, hovership.default_initial_state)

    def test_custom_creation(self):
        dynamics_parameters = {
            'ground_gravity': 0,
            'gravity_gradient': 0,
            'max_altitude': 4,
            'max_thrust': 4
        }
        default_initial_state = atleast_1d(2)
        hovership = DiscreteHovership(
            random_start=True,
            default_initial_state=default_initial_state,
            dynamics_parameters=dynamics_parameters
        )
        hovership.reset(s=default_initial_state)
        self.routine(hovership, default_initial_state)

        for t in range(10):
            hovership.reset()
            if hovership.s[0] != default_initial_state[0]:
                break
        else:
            self.assertTrue(False)

        hovership.reset(s=default_initial_state)
        hovership.step(action=atleast_1d(0.))
        self.assertEqual(hovership.s[0], default_initial_state[0])

    def test_stateaction_space_stability(self):
        hovership = Hovership()
        for t in range(100):
            s, r, failed = hovership.step(atleast_1d(1.))
            self.assertTrue(s in hovership.stateaction_space.state_space)

        hovership.reset()
        for t in range(100):
            s, r, failed = hovership.step(atleast_1d(0.))
            self.assertTrue(s in hovership.stateaction_space.state_space)

    def routine(self, hovership, initial_state):
        self.assertTrue(not hovership.in_failure_state)
        self.assertTrue(hovership.feasible)
        self.assertTrue(not hovership.has_failed)
        self.assertEqual(hovership.default_initial_state[0], initial_state[0])
        self.assertEqual(hovership.s, hovership.default_initial_state)

        r = 0
        MAX_ITER = 1e2
        t = 0
        while r == 0 and t < MAX_ITER:
            s, r, failed = hovership.step(atleast_1d(
                hovership.action_space[-1]
            ))
            t += 1
        self.assertTrue(hovership.s[0] >= hovership.default_initial_state[0])
        self.assertTrue(not failed)
        self.assertEqual(r, 1)

        hovership.reset(atleast_1d(0))
        self.assertEqual(hovership.s, atleast_1d(0))
        self.assertTrue(hovership.has_failed)
        self.assertTrue(hovership.in_failure_state)

        hovership.step(atleast_1d(1))
        self.assertEqual(hovership.s, atleast_1d(0))
        self.assertTrue(hovership.has_failed)
        self.assertTrue(hovership.in_failure_state)
