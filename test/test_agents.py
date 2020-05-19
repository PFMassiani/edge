import unittest
import numpy as np

from edge.agent import SafetyLearner, QLearner
from edge.envs import Hovership


class ToyHovership(Hovership):
    def __init__(self):
        dynamics_parameters = {
            'control_frequency': 0.1,
            'ground_gravity': 1,
            'shape': (2, 2)
        }
        super(ToyHovership, self).__init__(
            random_start=True,
            dynamics_parameters=dynamics_parameters
        )


class TestQLearner(unittest.TestCase):
    def test_qlearner(self):
        env = ToyHovership()
        hyperparameters = {
            'outputscale_prior': (1, 0.1),
            'lengthscale_prior': (0.2, 0.05),
            'noise_prior': (0.001, 0.001)
        }
        x_seed = np.array([1., 1.])
        y_seed = np.array([1.])

        agent = QLearner(
            env=env,
            greed=0.1,
            step_size=0.9,
            discount_rate=0.9,
            x_seed=x_seed,
            y_seed=y_seed,
            gp_params=hyperparameters,
            keep_seed_in_data=True
        )

        for episode in range(3):
            agent.reset()
            failed = agent.failed
            n_steps = 0
            while not failed and n_steps < 50:
                new_state, reward, failed = agent.step()
                n_steps += 1

        q_values = agent.Q_model[:, :].reshape(
            env.stateaction_space.shape
        )
        max_q = np.max(q_values)
        test = q_values < max_q * 0.6
        test[1, 1] = q_values[1, 1] == max_q
        self.assertTrue(np.all(test), 'The final values are not the expected '
                                      f'ones. Final values:\n{q_values}')


class TestSafetyLearner(unittest.TestCase):
    def __get_agent(self, gamma_cautious=0.9):
        env = ToyHovership()
        hyperparameters = {
            'outputscale_prior': (1, 0.1),
            'lengthscale_prior': (0.1, 0.05),
            'noise_prior': (0.001, 0.001)
        }
        x_seed = np.array([2., 1.])
        y_seed = np.array([1.])

        agent = SafetyLearner(
            env=env,
            gamma_optimistic=0.9,
            gamma_cautious=gamma_cautious,
            lambda_cautious=0,
            x_seed=x_seed,
            y_seed=y_seed,
            gp_params=hyperparameters,
        )
        return agent, env

    def test_safety_learner(self):
        agent, env = self.__get_agent()

        for episode in range(3):
            agent.reset()
            failed = agent.failed
            n_steps = 0
            while not failed and n_steps < 50:
                new_state, reward, failed = agent.step()
                n_steps += 1

        safety_values = agent.safety_model[:, :].reshape(
            env.stateaction_space.shape
        )
        max_s = np.max(safety_values)
        test = safety_values < 1e-5
        test[1, 1] = safety_values[1, 1] == max_s
        self.assertTrue(np.all(test), 'The final safety is not the expected '
                                      f'one. Final safety:\n{safety_values}')

    def test_safe_random_sampling(self):
        x_seed = np.array([1.45, 0.5])
        y_seed = np.array([.8])

        dynamics_parameters = {
            'shape': (50, 151)
        }
        self.env = Hovership(
            random_start=False,
            dynamics_parameters=dynamics_parameters,
            default_initial_state=x_seed[:1]
        )
        self.hyperparameters = {
            'outputscale_prior': (0.4, 2),
            'lengthscale_prior': (0.2, 0.2),
            'noise_prior': (0.001, 0.002)
        }
        self.agent = SafetyLearner(
            env=self.env,
            gamma_optimistic=0.6,
            gamma_cautious=0.95,
            lambda_cautious=0,
            x_seed=x_seed,
            y_seed=y_seed,
            gp_params=self.hyperparameters,
        )

        safe_state = self.agent.get_random_safe_state()
        self.assertTrue(safe_state is not None, 'No safe state found')
        measure = self.agent.safety_model.measure(
            state=safe_state,
            lambda_threshold=0,
            gamma_threshold=self.agent.safety_model.gamma_measure
        )

        self.assertTrue(
            measure > 0,
            f'The measure of state {safe_state} is {measure} and should be > 0'
        )
