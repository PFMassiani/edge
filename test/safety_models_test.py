import unittest
import numpy as np
import tempfile

from edge.envs import Hovership
from edge.model.safety_models import MaternSafety


class TestHovership(Hovership):
    def __init__(self):
        super(TestHovership, self).__init__(
            random_start=True,
            dynamics_parameters={
                'control_frequency': 0.1,
                'ground_gravity': 0.1,
                'shape': (2, 2)
            }
        )


class TestMeasure(MaternSafety):
    def __init__(self, env, gamma_optimistic, x_seed, y_seed):
        hyperparameters = {
            'outputscale_prior': (1, 0.1),
            'lengthscale_prior': (0.1, 0.05),
            'noise_prior': (0.001, 0.001)
        }
        super(TestMeasure, self).__init__(env, gamma_optimistic,
                                          x_seed, y_seed,
                                          gp_params=hyperparameters)


class TestSafetyMeasure(unittest.TestCase):
    def test_convergence(self):
        tol = 1e-5
        env = TestHovership()

        x_seed = np.array([1.8, 0.8])
        y_seed = np.array([1.])

        gamma = 0.1

        measure = TestMeasure(env=env, gamma_optimistic=gamma, x_seed=x_seed,
                              y_seed=y_seed)

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
                    state, 0.05, gamma, return_covar=True
                )
                if not cautious_actions.any():
                    raise NotImplementedError('Please implement the case where'
                                              ' no cautious action exists')
                else:
                    cautious_indexes = np.argwhere(cautious_actions)
                    most_variance_action = np.argmax(
                        covar_slice[cautious_actions]
                    )
                    action = tuple(cautious_indexes[most_variance_action])
                    action = env.action_space[action]
                new_state, reward, failed = env.step(action)
                measure.update(state, action, new_state, reward, failed)
                state = new_state
                failed = env.has_failed
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

    def test_level_set_shape(self):
        env = TestHovership()

        x_seed = np.array([1., 1.])
        y_seed = np.array([1.])

        gamma = 0.1

        measure = TestMeasure(env=env, gamma_optimistic=gamma, x_seed=x_seed,
                              y_seed=y_seed)

        level_set = measure.level_set(None, 0, gamma)
        self.assertEqual(
            level_set.shape[0],
            np.prod(env.stateaction_space.shape),
            'The level set does not have the right number of elements. '
            f'Expected number: {np.prod(env.stateaction_space.shape)} - '
            f'Actual shape: {level_set.shape}')

        meas = measure.measure(None, 0, gamma)
        self.assertEqual(
            meas.shape[0],
            np.prod(env.state_space.shape),
            'The measure does not have the expected number of elements. '
            f'Expected number: {np.prod(env.state_space.shape)} - '
            f'Actual shape: {meas.shape}')

    def test_save_load(self):
        env = Hovership()
        x_seed = np.array([1.45, 0.6])
        y_seed = np.array([0.8])
        x_blank = np.array([0., 0])
        y_blank = np.array([0.])
        hyperparameters = {
            'outputscale_prior': (0.4, 2),
            'lengthscale_prior': (0.2, 0.2),
            'noise_prior': (0.001, 0.002)
        }
        safety = MaternSafety(env, 0.7, x_seed, y_seed, hyperparameters)

        tmpdir = 'results/'#tempfile.TemporaryDirectory().name
        safety.save(tmpdir)
        safety.save_samples(tmpdir + 'samples.npz')

        blank = MaternSafety.load(tmpdir, env, 0.7, x_blank, y_blank)
        blank.load_samples(tmpdir + 'samples.npz')

        self.assertTrue((blank.gp.train_x == safety.gp.train_x).all())
        self.assertEqual(blank.gp.structure_dict, safety.gp.structure_dict)

