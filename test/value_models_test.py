import unittest
import numpy as np

from edge.model.value_models import QLearning
from edge.envs import Hovership
from edge.space import StateActionSpace, Box
from edge.reward import ConstantReward


class TopHovership(Hovership):
    def __init__(self):
        super(TopHovership, self).__init__(dynamics_parameters={
            'shape': (10, 3)
        })
        rewarded = StateActionSpace.from_product(Box([0.99, 0], [1, 1], (10,3)))
        reward = ConstantReward(
            self.stateaction_space,
            constant=1,
            rewarded_set=rewarded
        )
        self.reward = reward


class TestQLearning(unittest.TestCase):
    def test_policy_convergence(self):
        env = TopHovership()
        qlearning = QLearning(env.stateaction_space, 0.9, 0.9)

        def policy_from_q_values(q_values, eps=0.1):
            policy = np.ones_like(q_values) * eps / q_values.shape[1]
            for i, state in iter(env.state_space):
                policy[i, np.argmax(q_values[i, :])] += 1 - eps
            return policy

        eps = 0.1
        nA = qlearning.q_values.shape[1]
        for episode in range(10):
            print(f'Episode #{episode}')
            state = env.reset()
            stateindex = env.state_space.get_index_of(state, around_ok=True)
            state = env.state_space[stateindex]
            failed = False
            n_steps = 0
            while not failed and n_steps < 300:
                probas = np.ones(nA) * eps / nA
                probas[np.argmax(qlearning.q_values[stateindex, :])] += 1 - eps
                actionindex = np.random.choice(nA, p=probas)
                action = env.action_space[actionindex]
                new_state, reward, failed = env.step(action)
                new_index = env.state_space.get_index_of(new_state, around_ok=True)
                new_state = env.state_space[new_index]
                qlearning.update(state, action, new_state, reward)
                state = new_state
                stateindex = new_index
                n_steps += 1
            print(f'Completed in {n_steps}')
            print('Values:')
            print(qlearning.q_values)

        policy = policy_from_q_values(qlearning.q_values)
        print('=== Final policy')
        print(policy)
        self.assertTrue(True)
