import unittest
import warnings
import numpy as np

import safety_gym
from safety_gym.envs.engine import Engine
import gym
import gym.spaces as spaces
from gym.envs.registration import register

from edge.gym_wrappers import BoxWrapper, DiscreteWrapper, GymEnvironmentWrapper
from edge.agent import RandomAgent

class SpaceWrappers(unittest.TestCase):
    def test_box_wrapper(self):
        warnings.filterwarnings('ignore')
        gb = spaces.Box(0, 1, (2,2))
        eb = BoxWrapper(gb, (10,10,10,10))

        eelem = eb.sample()
        gelem = gb.sample()
        self.assertEqual(eelem.shape, (4,))
        self.assertEqual(eb.to_gym((2,3,4,5)).shape, gb.shape)
        self.assertEqual(eb.from_gym(gelem).shape, (4,))

        gb = spaces.Box(np.array([0,1]), np.array([2,3]))
        eb = BoxWrapper(gb, (10, 10))

        eelem = eb.sample()
        gelem = gb.sample()
        self.assertEqual(eelem.shape, (2,))
        self.assertEqual(eb.to_gym((2,3)).shape, gb.shape)
        self.assertEqual(eb.from_gym(gelem).shape, (2,))

        gb = spaces.Box(-np.inf, np.inf, (1,))
        eb = BoxWrapper(gb, (10, ), inf_ceiling=5)
        for t in range(100):
            eelem = eb.sample()
            self.assertTrue(np.abs(eelem)[0] <= 5)
            self.assertTrue(eelem in eb)

    def test_discrete_wrapper(self):
        gd = spaces.Discrete(10)
        ed = DiscreteWrapper(gd)

        g = gd.sample()
        e = ed.sample()
        self.assertEqual(ed.to_gym(e), int(e))
        self.assertEqual(ed.from_gym(g), g)


class SafetyGymEnvironmentWrappers(unittest.TestCase):
    def test_safety_gym_environment_creation(self):
        senv = gym.make('Safexp-PointGoal1-v0')
        env = GymEnvironmentWrapper(senv)

        config = {
            'robot_base': 'xmls/car.xml',
            'task': 'push',
            'observe_goal_lidar': True,
            'observe_box_lidar': True,
            'observe_hazards': True,
            'observe_vases': True,
            'constrain_hazards': True,
            'lidar_max_dist': 3,
            'lidar_num_bins': 16,
            'hazards_num': 4,
            'vases_num': 4
        }

        senv = Engine(config)
        register(id='SafexpTestEnvironment-v0',
                 entry_point='safety_gym.envs.mujoco:Engine',
                 kwargs={'config': config})
        env = GymEnvironmentWrapper(senv, failure_critical=True)

    def test_safety_gym_random_agent(self):
        senv = gym.make('Safexp-PointGoal1-v0')
        env = GymEnvironmentWrapper(senv)
        random_agent = RandomAgent(env)

        ep_ret, ep_cost = 0, 0
        for t in range(1000):
            new_state, reward, failed = random_agent.step()
            ep_ret += reward
            ep_cost += env.info.get('cost', 0)
            env.gym_env.render()
            if env.done:
                print('Episode Return: %.3f \t Episode Cost: %.3f' % (ep_ret, ep_cost))
                ep_ret, ep_cost = 0, 0
                random_agent.reset()


class GymEnvironmentWrappers(unittest.TestCase):
    def test_gym_environment_creation(self):
        gymenv = gym.make('LunarLander-v2')
        env = GymEnvironmentWrapper(gymenv)

        env = GymEnvironmentWrapper(gymenv, failure_critical=True)
        self.assertTrue(True)

    def test_gym_random_agent(self):
        gymenv = gym.make('LunarLander-v2')
        env = GymEnvironmentWrapper(gymenv)
        random_agent = RandomAgent(env)

        ep_ret, ep_cost = 0, 0
        for t in range(1000):
            new_state, reward, failed = random_agent.step()
            ep_ret += reward
            ep_cost += env.info.get('cost', 0)
            env.gym_env.render()
            if env.done:
                print('Episode Return: %.3f \t Episode Cost: %.3f' % (ep_ret, ep_cost))
                ep_ret, ep_cost = 0, 0
                random_agent.reset()


if __name__ == '__main__':
    unittest.main()
