import unittest
import numpy as np
from pathlib import Path

from edge import Simulation
from edge.agent import SafetyLearner
from edge.envs import Hovership
from edge.graphics.plotter import CoRLPlotter


class ToySimulation(Simulation):
    def __init__(self, output_directory, name, episodes=5):
        dynamics_parameters = {
            'shape': (10, 10)
        }
        self.env = Hovership(
            random_start=True,
            dynamics_parameters=dynamics_parameters
        )

        hyperparameters = {
            'outputscale_prior': (1, 0.1),
            'lengthscale_prior': (0.1, 0.05),
            'noise_prior': (0.001, 0.001)
        }
        x_seed = np.array([1., 1.])
        y_seed = np.array([1.])

        self.agent = SafetyLearner(
            env=self.env,
            gamma_optimistic=0.9,
            gamma_cautious=0.9,
            lambda_cautious=0.9,
            x_seed=x_seed,
            y_seed=y_seed,
            gp_params=hyperparameters,
        )

        plotters = {
            'Safety': CoRLPlotter(self.agent)
        }

        super(ToySimulation, self).__init__(output_directory, name, plotters)

        self.episodes = 5

    def run(self):
        for episode in range(self.episodes):
            self.agent.reset()
            failed = self.agent.failed
            n_steps = 0
            while not failed and n_steps < 50:
                new_state, reward, failed = self.agent.step()
                n_steps += 1
            self.save_figs(prefix=f'episode_{episode}')

        self.compile_gif()


class TestSimulation(unittest.TestCase):
    def test_run(self):
        sim = ToySimulation(
            output_directory='results',
            name='foo',
            episodes=1
        )

        sim.run()

        paths = [
            'results',
            'results/foo',
            'results/foo/figs',
            'results/foo/logs',
            'results/foo/figs/episode_0_Safety.pdf',
            'results/foo/figs/Safety.gif',
        ]

        paths = [Path(p) for p in paths]

        for p in paths:
            self.assertTrue(p.exists(), f'Path {str(p)} does not exist')
