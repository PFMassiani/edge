import unittest
from pathlib import Path

import safety_gym
import gym

from edge.gym_wrappers import GymEnvironmentWrapper
from edge.agent.policy_learner import PolicyLearner

class MLPTests(unittest.TestCase):
    def test_random_agent(self):
        senv = gym.make('Safexp-PointGoal1-v0')
        env = GymEnvironmentWrapper(senv)
        net_path = Path('/home/ics/massiani/Desktop/MPI_ln/WD/edge/data/models/ppo_lagrangian_long_train')
        random_agent = PolicyLearner(env, net_path)

        ep_ret, ep_cost = 0, 0
        for t in range(10000):
            new_state, reward, failed = random_agent.step()
            ep_ret += reward
            ep_cost += env.info.get('cost', 0)
            env.gym_env.render()
            if env.done:
                print('Episode Return: %.3f \t Episode Cost: %.3f' % (ep_ret, ep_cost))
                ep_ret, ep_cost = 0, 0
                random_agent.reset()