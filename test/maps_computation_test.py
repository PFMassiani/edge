import unittest
import numpy as np

from edge.envs import Hovership, DiscreteHovership


class MyDiscreteHovership(DiscreteHovership):
    def __init__(self):
        dynamics_parameters = {
            'ground_gravity': 1,
            'gravity_gradient': 1,
            'max_thrust': 3,
            'max_altitude': 5,
            'minimum_gravity_altitude': 4
        }
        super(MyDiscreteHovership, self).__init__(
            self, dynamics_parameters=dynamics_parameters
        )

class MyTestCase(unittest.TestCase):
    def test_dynamics_map(self):
        env = MyDiscreteHovership()
        Q_map = env.dynamics.compute_map()

        true_Q_map = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 2],
            [1, 2, 3, 4],
            [3, 4, 5, 5],
            [4, 5, 5, 5]
        ])

        self.assertTrue(
            (true_Q_map == Q_map).all(),
            'Error: computed Q_map is different from ground truth.\nComputed:\n'
            f'{Q_map}\nGround truth:\n{true_Q_map}'
        )


if __name__ == '__main__':
    unittest.main()
