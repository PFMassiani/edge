import unittest
import numpy as np
from pathlib import Path

from edge import Simulation
from edge.agent import SafetyLearner
from edge.envs import Hovership
from edge.model.safety_models import SafetyTruth
from edge.graphics.plotter import CoRLPlotter


class ToySimulation(Simulation):
    def __init__(self, output_directory, name, max_samples=250,
                 gamma_optimistic=0.9, gamma_cautious=0.9,
                 lambda_cautious=0.1, lengthscale_prior=(0.1, 0.05),
                 shape=(10, 10), hyperparameters=None, ground_truth=None,
                 every=50):
        x_seed = np.array([1.45, 0.5])
        y_seed = np.array([1.])

        dynamics_parameters = {
            'shape': shape
        }

        self.env = Hovership(
            random_start=False,
            dynamics_parameters=dynamics_parameters,
            default_initial_state=x_seed[:1]
        )

        if hyperparameters is None:
            hyperparameters = {}
        default_hyperparameters = {
            'outputscale_prior': (1, 0.1),
            'lengthscale_prior': lengthscale_prior,
            'noise_prior': (0.001, 0.001)
        }
        default_hyperparameters.update(hyperparameters)
        hyperparameters = default_hyperparameters

        if ground_truth is None:
            self.ground_truth = None
        else:
            self.ground_truth = SafetyTruth(self.env)
            self.ground_truth.from_vibly_file(ground_truth)

        self.agent = SafetyLearner(
            env=self.env,
            gamma_optimistic=gamma_optimistic,
            gamma_cautious=gamma_cautious,
            lambda_cautious=lambda_cautious,
            x_seed=x_seed,
            y_seed=y_seed,
            gp_params=hyperparameters,
        )

        self.agent.reset()

        plotters = {
            'Safety': CoRLPlotter(self.agent, self.ground_truth)
        }

        super(ToySimulation, self).__init__(output_directory, name, plotters)

        self.max_samples = max_samples
        self.every = every

    def run(self):
        n_samples = 0
        while n_samples < self.max_samples:
            self.agent.reset()
            failed = self.agent.failed
            n_steps = 0
            while not failed and n_steps < 50:
                if n_samples % self.every == 0:
                    self.save_figs(prefix=f'{n_samples}')

                n_samples += 1
                n_steps += 1
                old_state = self.agent.state
                new_state, reward, failed = self.agent.step()

                action = self.agent.last_action
                print(f'Step {n_samples}/{self.max_samples} - {old_state} '
                      f' -> {action} -> {new_state} ({failed})')

                self.on_run_iteration(old_state, action, new_state, reward,
                                      failed)
                if n_samples >= self.max_samples:
                    break

        self.compile_gif()


class TestSimulation(unittest.TestCase):
    def test_run(self):
        sim = ToySimulation(
            output_directory='results',
            name='foo',
            max_samples=100,
            shape=(11, 11)
        )

        sim.run()

        paths = [
            'results',
            'results/foo',
            'results/foo/figs',
            'results/foo/logs',
            'results/foo/figs/0_Safety.pdf',
            'results/foo/figs/Safety.gif',
        ]

        paths = [Path(p) for p in paths]

        for p in paths:
            self.assertTrue(p.exists(), f'Path {str(p)} does not exist')

    def test_optimistic_init(self):
        sim = ToySimulation(
            output_directory='results',
            name='test_optimistic_init',
            max_samples=300,
            gamma_optimistic=0.5,
            gamma_cautious=0.8,
            lambda_cautious=0.1,
            lengthscale_prior=(0.3, 0.01),
            shape=(101, 101),
            ground_truth='../vibly/data/dynamics/hover_map.pickle',
            every=50
            # hyperparameters={
            #     'hyperparameters_initialization': {
            #         'mean_module.constant': 0.
            #     }
            # }
        )

        sim.run()

        final_measure = sim.agent.safety_model[:, :]
        self.assertTrue(
            (final_measure > 1 - 1e-3).all(),
            'Final measure is not the expected one. Final measure: '
            f'{final_measure}'
        )
