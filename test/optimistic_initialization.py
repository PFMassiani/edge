import numpy as np
from scipy.stats import norm

from edge import Simulation
from edge.agent import SafetyLearner
from edge.envs import Hovership
from edge.model.safety_models import SafetyTruth, MaternSafety
from edge.graphics.plotter import SafetyPlotter, DetailedSafetyPlotter


class OptimisticSimulation(Simulation):
    def __init__(self, max_samples, gamma_optimistic, gamma_cautious,
                 lambda_cautious, shape, every):
        self.x_seed = np.array([1.45, 0.5])
        self.y_seed = np.array([.8])
        dynamics_parameters = {
            'shape': shape
        }
        self.env = Hovership(
            random_start=True,
            dynamics_parameters=dynamics_parameters,
            default_initial_state=self.x_seed[:1]
        )

        self.ground_truth = SafetyTruth(self.env)
        self.ground_truth.from_vibly_file(
            '../data/ground_truth/from_vibly/hover_map.pickle'
        )

        self.hyperparameters = {
            'outputscale_prior': (0.4, 2),
            'lengthscale_prior': (0.1, 0.1),
            'noise_prior': (0.001, 0.002)
        }
        self.agent = SafetyLearner(
            env=self.env,
            gamma_optimistic=gamma_optimistic,
            gamma_cautious=gamma_cautious,
            lambda_cautious=lambda_cautious,
            x_seed=self.x_seed,
            y_seed=self.y_seed,
            gp_params=self.hyperparameters,
        )
        plotters = {
            'DetailedSafety': DetailedSafetyPlotter(self.agent, self.ground_truth)
        }

        super(OptimisticSimulation, self).__init__(
            'results', 'optimistic', plotters
        )

        self.max_samples = max_samples
        self.every = every
        self.samples_path = self.output_directory / 'samples'
        self.samples_path.mkdir(parents=True, exist_ok=True)
        self.model_path = self.output_directory / 'model'
        self.model_path.mkdir(parents=True, exist_ok=True)

        failure_indexes = np.argwhere(self.ground_truth.failure_set == 1)
        self.failure_set = np.array([
            self.ground_truth.stateaction_space[tuple(index)]
            for index in failure_indexes[::3]
        ])

    def run_optim(self):
        train_x, train_y = self.ground_truth.get_training_examples(
            n_examples=2000,
            from_viable=True,
            from_failure=False
        )
        self.agent.fit_models(train_x, train_y, epochs=20)

    def save_samples(self, name):
        self.agent.safety_model.save_samples(str(self.samples_path / name))

    def load_samples(self, name):
        self.agent.safety_model.load_samples(str(self.samples_path / name))

    def save_model(self):
        self.agent.safety_model.save(str(self.model_path))

    def load_model(self):
        self.agent.safety_model = MaternSafety.load(str(self.model_path),
            self.env, self.agent.safety_model.gamma_measure,
            self.x_seed, self.y_seed
        )

    def check_failure_set(self):
        model = self.agent.safety_model

        measure_slice, covar_slice = model._query(
            self.failure_set, return_covar=True)
        level_value = norm.cdf(
                (measure_slice - 0) / np.sqrt(covar_slice)
            )
        failure_levels = level_value > model.gamma_measure

        if failure_levels.any():
            print('Nonzero value in the failure set !')

    def run_learning(self):
        n_samples = 0
        self.save_figs(prefix='0')
        while n_samples < self.max_samples:
            failed = self.agent.failed
            n_steps = 0
            while not failed and n_steps < 50:
                #self.check_failure_set()
                n_samples += 1
                n_steps += 1
                old_state = self.agent.state
                new_state, reward, failed, _ = self.agent.step()
                action = self.agent.last_action

                self.on_run_iteration(
                    n_samples,
                    old_state, action, new_state, reward, failed
                )

                if n_samples >= self.max_samples:
                    break

            reset_state = self.agent.get_random_safe_state()
            if reset_state is None:
                raise Exception('The whole measure is 0. There is no safe '
                                'action.')
            self.agent.reset(reset_state)

    def on_run_iteration(self, n_samples, old_state, action, new_state,
                         reward, failed):
        super(OptimisticSimulation, self).on_run_iteration(
            old_state, action, new_state, reward, failed
        )
        print(f'Step {n_samples}/{self.max_samples} - {old_state} '
              f' -> {action} -> {new_state} ({failed})')
        if n_samples % self.every == 0:
            self.save_figs(prefix=f'{n_samples}')


if __name__ == '__main__':
    np.random.seed(2)
    sim = OptimisticSimulation(
        max_samples=500,
        gamma_optimistic=0.52,
        gamma_cautious=0.8,
        lambda_cautious=0.0,
        shape=(201, 151),
        every=500
    )
    optimize = True
    load_hyper = False
    samples_to_load = None #'1500.npz'

    if optimize:
        print('optimizing hyperparameters...')
        sim.run_optim()
        print('saving hyperparameters...')
        sim.save_model()
    elif load_hyper:
        print('loading hyperparameters...')
        sim.load_model()
    else:
        print(f'keeping default hyperparameters: {sim.hyperparameters}... ')
    if samples_to_load is not None:
        print('loading samples...')
        sim.load_samples(samples_to_load)

    print('learning...')
    sim.run_learning()
    print('saving outputs...')
    sim.save_samples('last')
    sim.compile_gif()
