import unittest
import numpy as np

from edge.model.value_models import QLearning
from edge.envs import DiscreteHovership
from edge.space import Discrete, StateActionSpace
from edge.reward import ConstantReward


class ToyHovership(DiscreteHovership):
    def __init__(self):
        super(ToyHovership, self).__init__(default_initial_state=np.array([2.]))
        max_altitude = self.state_space.n - 1
        max_thrust = self.action_space.n -1
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


class TestQLearning(unittest.TestCase):
    def test_policy_convergence(self):
        env = ToyHovership()
        nA = env.action_space.n
        qlearning = QLearning(env.stateaction_space, 0.9, 0.9)

        def policy_from_q_values(q_values, eps=0.1):
            policy = np.ones_like(q_values) * eps / q_values.shape[1]
            for i, state in iter(env.state_space):
                policy[i, np.argmax(q_values[i, :])] += 1 - eps
            return policy

        eps = 0.1
        nA = qlearning.q_values.shape[1]
        for episode in range(100):
            reset_state = np.random.choice([2, 3])
            state = env.reset(reset_state)
            failed = False
            n_steps = 0
            while not failed and n_steps < 10:
                probas = np.ones(nA) * eps / nA
                probas[np.argmax(qlearning.q_values[state, :])] += 1 - eps
                action = env.action_space[np.random.choice(nA, p=probas)]
                new_state, reward, failed = env.step(action)
                qlearning.update(state, action, new_state, reward)
                state = new_state
                n_steps += 1

        policy = policy_from_q_values(qlearning.q_values, eps=0)

        self.assertTrue(np.all(policy[2, :] == [0, 0, 1]), 'Final policy '
                                                           f'is\n{policy}')
