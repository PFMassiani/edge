import numpy as np
import unittest

from edge.envs.continuous_cartpole import ContinuousCartPole
from edge.utils.control import dlqr, lqr


class CartpoleLQRTest(unittest.TestCase):
    def test_custom_continuous_time_lqr(self):
        env = ContinuousCartPole(discretization_shape=(10, 10, 10, 10, 10))
        A, B = env.linearization(discrete_time=False)
        Q = 2 * np.eye(4)
        R = 0.4
        K, S, E = lqr(A, B, Q, R)
        print('Continuous time')
        print(f'Feedback gain: {K}')
        print(f'Riccati equation solution: {S}')
        print(f'Eigenvalues of the closed-loop system: {E}')
        self.assertTrue(True)

    def test_discrete_time_lqr(self):
        env = ContinuousCartPole(discretization_shape=(10, 10, 10, 10, 10))
        A, B = env.linearization()
        Q = np.eye(4)
        R = 10
        K, S, E = dlqr(A, B, Q, R)
        print('Discrete time')
        print(f'Feedback gain: {K}')
        print(f'Riccati equation solution: {S}')
        print(f'Eigenvalues of the closed-loop system: {E}')
        self.assertTrue(True)