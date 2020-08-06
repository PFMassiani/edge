import unittest
import numpy as np
import tempfile
import gym

from edge.envs import Hovership
from edge.gym_wrappers import GymEnvironmentWrapper
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


class LunarLander(GymEnvironmentWrapper):
    def __init__(self, discretization_shape):
        gym_env = gym.make('LunarLanderContinuous-v2')
        super(LunarLander, self).__init__(gym_env, discretization_shape)


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
                cautious_actions = cautious_actions.squeeze()
                covar_slice = covar_slice.squeeze()
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

    def test_level_set_shape_0(self):
        env = TestHovership()

        x_seed = np.array([1., 1.])
        y_seed = np.array([1.])

        gamma = 0.1

        measure = TestMeasure(env=env, gamma_optimistic=gamma, x_seed=x_seed,
                              y_seed=y_seed)

        def check_level_set_on_query(query, query_len):
            output_level_shape = (query_len, ) + env.action_space.shape
            output_measure_shape = (query_len,)
            level_set = measure.level_set(query, 0, gamma)
            self.assertEqual(
                level_set.shape,
                output_level_shape,
                'The level set does not have the right shape. '
                f'Expected shape: {output_level_shape} - '
                f'Actual shape: {level_set.shape}')

            meas = measure.measure(query, 0, gamma)
            self.assertEqual(
                meas.shape,
                output_measure_shape,
                'The measure does not have the expected shape. '
                f'Expected shape: {output_measure_shape} - '
                f'Actual shape: {meas.shape}')

        s_query = np.array([0.5])
        check_level_set_on_query(s_query, 1)
        s_query = slice(None, None, None)
        check_level_set_on_query(s_query, np.prod(env.state_space.shape))

    def test_level_set_shape_1(self):
        env = LunarLander(discretization_shape=tuple(10 for _ in range(8)))

        x_seed = np.array([0, 1.4, 0, 0, 0, 0, 0, 0, 1, 0])
        y_seed = np.array([1.])

        gamma = 0.1

        measure = TestMeasure(env=env, gamma_optimistic=gamma, x_seed=x_seed,
                              y_seed=y_seed)

        def check_level_set_on_query(query, query_len):
            output_level_shape = (query_len, ) + env.action_space.shape
            output_measure_shape = (query_len,)
            level_set = measure.level_set(query, 0, gamma)
            self.assertEqual(
                level_set.shape,
                output_level_shape,
                'The level set does not have the right shape. '
                f'Expected shape: {output_level_shape} - '
                f'Actual shape: {level_set.shape}')

            meas = measure.measure(query, 0, gamma)
            self.assertEqual(
                meas.shape,
                output_measure_shape,
                'The measure does not have the expected shape. '
                f'Expected shape: {output_measure_shape} - '
                f'Actual shape: {meas.shape}')

        s_query = np.array([0, 1.4, 0, 0, 0, 0, 0, 0])
        check_level_set_on_query(s_query, 1)
        s_query = (
            slice(0, 5, 1),
            np.array([1.4]),
            np.array([0]),
            np.array([0]),
            np.array([0]),
            np.array([0]),
            np.array([0]),
            np.array([0]),
        )
        check_level_set_on_query(s_query, 5)

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

