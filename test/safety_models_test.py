import unittest
import numpy as np

from edge.envs import Hovership
from edge.model.safety_models import MaternSafety


class TestHovership(Hovership):
    def __init__(self):
        super(TestHovership, self).__init__(
            random_start=True,
            dynamics_parameters={
                'control_frequency': 0.1,
                'ground_gravity': 1,
                'shape': (2, 2)
            }
        )


class TestMeasure(MaternSafety):
    def __init__(self, env, x_seed, y_seed):
        hyperparameters = {
            'outputscale_prior': (1, 0.1),
            'lengthscale_prior': (0.1, 0.05),
            'noise_prior': (0.001, 0.001)
        }
        super(TestMeasure, self).__init__(env, x_seed, y_seed,
                                          **hyperparameters)


class TestSafetyMeasure(unittest.TestCase):
    def test_convergence(self):
        tol = 1e-1
        env = TestHovership()

        x_seed = np.array([1., 1.])
        y_seed = np.array([1.])

        measure = TestMeasure(env=env, x_seed=x_seed, y_seed=y_seed)

        epochs = 3
        max_steps = 100
        for episode in range(epochs):
            failed = True
            while failed:
                state = env.reset(s=x_seed[:1])
                failed = env.has_failed
            n_steps = 0
            while not failed and n_steps < max_steps:
                cautious_actions, covar_slice = measure.level_set(
                    state, 0.05, 0.1, return_covar=True
                )
                if not cautious_actions.any():
                    raise NotImplementedError('Please implement the case where'
                                              ' no cautious action exists')
                else:
                    cautious_indexes = np.nonzero(cautious_actions)[0]
                    most_variance_action = np.argmax(
                        covar_slice[cautious_actions]
                    )
                    action = cautious_indexes[most_variance_action]
                    action = env.action_space[action]
                new_state, reward, failed = env.step(action)
                measure.update(state, action, new_state, reward, failed)
                state = new_state
                n_steps += 1

        final_measure = measure[:, :].reshape(
            env.stateaction_space.shape
        )
        expected_final = np.array([[0, 0], [0, 1]]).astype(np.bool)
        self.assertTrue(
            np.all((final_measure > tol) == expected_final),
            f'Final measure does not match the expected one. Final measure :\n'
            f'{final_measure}\nExpected final measure:\n{expected_final}'
        )
