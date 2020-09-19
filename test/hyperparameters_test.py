import unittest
import numpy as np

from edge import Simulation
from edge.agent import SafetyLearner
from edge.envs import Hovership
from edge.model.safety_models import SafetyTruth
from edge.graphics.plotter import SafetyPlotter


class HyperparametersSimulation(Simulation):
    def __init__(self, output_directory, name, max_samples,
                 gamma_optimistic, gamma_cautious, lambda_cautious,
                 shape, ground_truth,
                 random_start=False, every=50):
        x_seed = np.array([1.45, 0.5])
        y_seed = np.array([.8])

        dynamics_parameters = {
            'shape': shape
        }
        self.env = Hovership(
            random_start=random_start,
            dynamics_parameters=dynamics_parameters,
            default_initial_state=x_seed[:1]
        )

        self.ground_truth = SafetyTruth(self.env)
        self.ground_truth.from_vibly_file(ground_truth)

        self.hyperparameters = {
            'outputscale_prior': (0.4, 2),
            'lengthscale_prior': (0.2, 0.2),
            'noise_prior': (0.001, 0.002)
        }
        self.agent = SafetyLearner(
            env=self.env,
            gamma_optimistic=gamma_optimistic,
            gamma_cautious=gamma_cautious,
            lambda_cautious=lambda_cautious,
            x_seed=x_seed,
            y_seed=y_seed,
            gp_params=self.hyperparameters,
        )
        self.agent.reset()

        plotters = {
            'Safety': SafetyPlotter(self.agent, self.ground_truth)
        }

        super(HyperparametersSimulation, self).__init__(
            output_directory, name, plotters
        )

        self.max_samples = max_samples
        self.every = every
        self.random_start = random_start

    def run(self):
        self.run_optim()
        self.run_learning()

    def run_optim(self):
        train_x, train_y = self.ground_truth.get_training_examples(
            n_examples=2000,
            from_viable=True,
            from_failure=False
        )
        self.agent.fit_models(train_x, train_y, epochs=20)

    def run_learning(self):
        gamma_optim_increment = (
            self.agent.gamma_cautious - self.agent.safety_model.gamma_measure
        ) / self.max_samples
        n_samples = 0
        self.save_figs(prefix='0')
        while n_samples < self.max_samples:
            failed = self.agent.failed
            n_steps = 0
            while not failed and n_steps < 50:
                n_samples += 1
                n_steps += 1
                old_state = self.agent.state
                new_state, reward, failed = self.agent.step()
                action = self.agent.last_action

                self.on_run_iteration(
                    n_samples,
                    old_state, action, new_state, reward, failed
                )

                if n_samples >= self.max_samples:
                    break
            if self.random_start:
                reset_state = np.atleast_1d(
                    np.random.choice(np.linspace(0, 1.5, 100))
                )
                self.agent.reset(reset_state)
            else:
                reset_state = self.agent.get_random_safe_state()
                if reset_state is None:
                    raise Exception('The whole measure is 0. There is no safe '
                                    'action.')
                self.agent.reset(reset_state)

            self.agent.safety_model.gamma_measure += gamma_optim_increment

        self.compile_gif()

    def on_run_iteration(self, n_samples, old_state, action, new_state,
                         reward, failed):
        super(HyperparametersSimulation, self).on_run_iteration(
            old_state, action, new_state, reward, failed
        )
        print(f'Step {n_samples}/{self.max_samples} - {old_state} '
              f' -> {action} -> {new_state} ({failed})')
        if n_samples % self.every == 0:
            self.save_figs(prefix=f'{n_samples}')


class TestHyperparametersLearning(unittest.TestCase):
    def __get_simulation(self, name, max_samples=250, every=50):
        return HyperparametersSimulation(
            output_directory='results/',
            name=name,
            max_samples=max_samples,
            gamma_optimistic=0.6,
            gamma_cautious=0.95,
            lambda_cautious=0.,
            shape=(201, 151),
            ground_truth='../data/ground_truth/from_vibly/hover_map.pickle',
            every=every
        )

    def __print_hyperparameters(self, sim):
        gp = sim.agent.safety_model.gp
        print('Outputscale:', gp.covar_module.outputscale)
        print('Lengthscale:', gp.covar_module.base_kernel.lengthscale)
        print('Noise:', gp.likelihood.noise)

    def test_optimization(self):
        sim = self.__get_simulation('test_optimization')
        sim.run_optim()

        outputscale = sim.agent.safety_model.gp.covar_module.outputscale
        lengthscale = sim.agent.safety_model.gp.covar_module.\
            base_kernel.lengthscale
        # noise = sim.agent.safety_model.gp.likelihood.noise
        self.assertTrue(
            sim.hyperparameters['outputscale_prior'][0] != outputscale,
            f'The outputscale has not changed: {outputscale}'
        )
        self.assertTrue(
            (sim.hyperparameters['lengthscale_prior'][0] != lengthscale).all(),
            f'The lengthscale has not changed: {lengthscale}'
        )
        self.__print_hyperparameters(sim)

    def test_full_learning(self):
        sim = self.__get_simulation(
            'test_full_learning', max_samples=300, every=50
        )
        sim.run_optim()
        self.__print_hyperparameters(sim)
        sim.run_learning()

    def test_optimistic_init(self):
        sim = HyperparametersSimulation(
            output_directory='results/',
            name='test_optimistic_init',
            max_samples=500,
            gamma_optimistic=0.52,
            gamma_cautious=0.7,
            lambda_cautious=0.1,
            shape=(201, 151),
            ground_truth='../data/ground_truth/from_vibly/hover_map.pickle',
            random_start=False,
            every=50
        )
        sim.run()
