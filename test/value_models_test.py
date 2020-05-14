import unittest
import numpy as np

from edge.model.value_models import QLearning, GPQLearning
from edge.envs import DiscreteHovership, Hovership
from edge.space import Discrete, StateActionSpace
from edge.reward import ConstantReward


class ToyHovership(DiscreteHovership):
    def __init__(self):
        super(ToyHovership, self).__init__(
            default_initial_state=np.array([2.]))
        max_altitude = self.state_space.n - 1
        max_thrust = self.action_space.n - 1
        rewarded_set = StateActionSpace(
            Discrete(n=2, start=max_altitude - 1, end=max_altitude),
            Discrete(max_thrust + 1)
        )
        reward = ConstantReward(
            self.stateaction_space,
            constant=1,
            rewarded_set=rewarded_set
        )
        self.reward = reward


# class TestQLearning(unittest.TestCase):
#     def test_policy_convergence(self):
#         env = ToyHovership()
#         nA = env.action_space.n
#         qlearning = QLearning(env, 0.9, 0.9)
#
#         def policy_from_q_values(q_values, eps=0.1):
#             policy = np.ones_like(q_values) * eps / q_values.shape[1]
#             for i, state in iter(env.state_space):
#                 policy[i, np.argmax(q_values[i, :])] += 1 - eps
#             return policy
#
#         eps = 0.1
#         nA = qlearning.q_values.shape[1]
#         for episode in range(100):
#             reset_state = np.random.choice([2, 3])
#             state = env.reset(reset_state)
#             failed = False
#             n_steps = 0
#             while not failed and n_steps < 10:
#                 probas = np.ones(nA) * eps / nA
#                 probas[np.argmax(qlearning.q_values[state, :])] += 1 - eps
#                 action = env.action_space[np.random.choice(nA, p=probas)]
#                 new_state, reward, failed = env.step(action)
#                 qlearning.update(state, action, new_state, reward)
#                 state = new_state
#                 n_steps += 1
#
#         policy = policy_from_q_values(qlearning.q_values, eps=0)
#
#         self.assertTrue(np.all(policy[2, :] == [0, 0, 1]), 'Final policy '
#                                                            f'is\n{policy}')


class TestGPQLearning(unittest.TestCase):
    def test_policy_convergence(self):
        hovership_params = {
            'shape': (100, 2)
        }
        env = Hovership(random_start=True,
                        dynamics_parameters=hovership_params)
        hyperparameters = {
            'outputscale_prior': (1, 0.1),
            'lengthscale_prior': (0.2, 0.05),
            'noise_prior': (0.001, 0.001)
        }
        x_seed = np.array([0.85, 1.])
        y_seed = np.array([1.])
        gpqlearning = GPQLearning(env, 0.9, 0.9, x_seed=x_seed, y_seed=y_seed,
                                  gp_params=hyperparameters)
        nA = env.action_space.index_shape[0]
        eps = 0.1
        for episode in range(3):
            state = env.reset()
            failed = env.has_failed
            n_steps = 0
            while not failed and n_steps < 100:
                probas = np.ones(nA) * eps / nA
                probas[np.argmax(gpqlearning[state, :])] += 1 - eps
                action = env.action_space[np.random.choice(nA, p=probas)]
                new_state, reward, failed = env.step(action)
                gpqlearning.update(state, action, new_state, reward)
                state = new_state
                n_steps += 1

        def policy_from_gpq(gpq):
            q_values = gpq[:, :].reshape(
                gpq.env.stateaction_space.index_shape
            )
            policy = np.zeros_like(q_values)
            for i, _ in iter(env.state_space):
                policy[i, np.argmax(q_values[i, :])] = 1
            return policy

        policy = policy_from_gpq(gpqlearning)
        print(policy)
        self.assertTrue(False, "The computation of the policy works, but "
                               "the convergence value is not tested.")

    def test_indexing(self):
        hovership_params = {
            'shape': (100, 5)
        }
        env = Hovership(random_start=True,
                        dynamics_parameters=hovership_params)
        x_seed = np.array([1., 1.])
        y_seed = np.array([1])
        hyperparameters = {
            'outputscale_prior': (1, 0.1),
            'lengthscale_prior': (0.2, 0.05),
            'noise_prior': (0.001, 0.001)
        }
        gpqlearning = GPQLearning(env, 0.9, 0.9, x_seed=x_seed, y_seed=y_seed,
                                  gp_params=hyperparameters)

        query = gpqlearning._get_query_from_index(
            (np.array([0.5]), slice(None, None, None))
        )
        self.assertEqual(query.shape, (5, 2))
        self.assertTrue(np.all(query[:, 0] == 0.5))

        pred = gpqlearning.gp.predict(query).mean.numpy()
        self.assertEqual(pred.shape, (5,))
