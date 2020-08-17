import numpy as np
import unittest
import control
import gym
import warnings

from edge.gym_wrappers import GymEnvironmentWrapper
from edge.utils.control import dlqr, lqr


class CartPole(GymEnvironmentWrapper):
    def __init__(self, discretization_shape=(10, 10, 10, 10, 10)):
        gym_env = gym.make('CartPole-v1')
        super(CartPole, self).__init__(gym_env, discretization_shape)

    @property
    def in_failure_state(self):
        # This condition is taken from OpenAI Gym documentation
        failed = np.abs(self.s[0]) > 2.4  # Linear position
        failed |= np.abs(self.s[2]) > 12 * np.pi / 180  # Angular position
        return failed

    def linearization(self, discrete_time=True):
        # The full equations are taken from
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        # which is the source given by OpenAI Gym for the dynamics
        g = self.gym_env.gravity
        m = self.gym_env.total_mass
        eta = self.gym_env.masspole / m
        l = self.gym_env.length * 2  # Gym stores half the pole's length
        a12 = g * eta / (eta - 4/3)
        a32 = (g / l) / (4/3 - eta)
        A = np.array([[0, 1, 0,   0],
                      [0, 0, a12, 0],
                      [0, 0, 0,   1],
                      [0, 0, a32, 0]])
        B = (1/m) * np.array([0,
                              (4/3) / (4/3 - eta),
                              0,
                              -1 / l / (4/3 - eta)]).reshape((-1, 1))
        # The action we give to Gym is simply +- 1: we need to multiply B
        # by the true magnitude of the force so we can feed it actions directly
        B = self.gym_env.force_mag * B

        if discrete_time:
            # A and B are the dynamics matrices of the _continuous_ dynamics.
            # We need to apply the integration scheme to get the true matrices
            integrator = self.gym_env.kinematics_integrator
            if integrator != 'euler':
                warnings.warn(f'Kinematics integrator is {integrator}, but only '
                              f'euler is supported.')

            tau = self.control_frequency if self.control_frequency is not None \
                else self.gym_env.tau
            A = np.eye(A.shape[0], dtype=float) + tau * A
            B = tau * B

        return A, B


class CartpoleLQRTest(unittest.TestCase):
    def test_continuous_time_lqr(self):
        env = CartPole()
        A, B = env.linearization(discrete_time=False)
        Q = 2 * np.eye(4)
        R = 0.4
        K, S, E = control.lqr(A, B, Q, R)
        print(f'Feedback gain: {K}')
        print(f'Riccati equation solution: {S}')
        print(f'Eigenvalues of the closed-loop system: {E}')
        self.assertTrue(True)

    def test_custom_continuous_time_lqr(self):
        env = CartPole()
        A, B = env.linearization(discrete_time=False)
        Q = 2 * np.eye(4)
        R = 0.4
        K, S, E = lqr(A, B, Q, R)
        print(f'Feedback gain: {K}')
        print(f'Riccati equation solution: {S}')
        print(f'Eigenvalues of the closed-loop system: {E}')
        self.assertTrue(True)

    def test_discrete_time_lqr(self):
        env = CartPole()
        A, B = env.linearization()
        Q = np.eye(4)
        R = 10
        K, S, E = dlqr(A, B, Q, R)
        print(f'Feedback gain: {K}')
        print(f'Riccati equation solution: {S}')
        print(f'Eigenvalues of the closed-loop system: {E}')
        self.assertTrue(True)