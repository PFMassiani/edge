import unittest
import numpy as np
import matplotlib.pyplot as plt

from edge.graphics.plotter import RewardFailurePlotter


class PlottersTest(unittest.TestCase):
    def test_reward_failure_plotter(self):
        agents_names = ['a', 'b']
        window_size = 10
        padding_value = 10
        plotter = RewardFailurePlotter(agents_names, window_size, padding_value)

        N_EPISODES = 10
        EP_LENGTH = 10
        rewards = {
            'a': [k*np.linspace(0, 5, EP_LENGTH) for k in range(N_EPISODES)],
            'b': [k*np.linspace(0, 5, EP_LENGTH)**2 for k in range(N_EPISODES)]
        }

        failures = {
            'a': [[0]*(EP_LENGTH-1) + [1]] * N_EPISODES,
            'b': [[0]*(EP_LENGTH-1) + [1]] * int(N_EPISODES/2) + \
                 [[0] * EP_LENGTH] * (N_EPISODES - int(N_EPISODES/2))
        }

        done = [False] * (EP_LENGTH - 1) + [True]
        # These are not used by the plotter: we can put whatever we want
        state = None
        new_state = None
        action = None

        for aname in agents_names:
            for n_episode in range(N_EPISODES):
                for n_step in range(EP_LENGTH):
                    plotter.on_run_iteration(
                        aname,
                        state=state,
                        action=action,
                        new_state=new_state,
                        reward=rewards[aname][n_episode][n_step],
                        failed=failures[aname][n_episode][n_step],
                        done=done[n_step]
                    )
        figure = plotter.get_figure()
        plt.show()
        self.assertTrue(True)