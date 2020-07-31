from pathlib import Path
import numpy as np

from edge import ModelLearningSimulation
from edge.graphics.plotter import SoftHardPlotter
from edge.model.safety_models import SafetyTruth

from soft_hard_parameterization import LowGoalSlip, SoftHardLearner


def affine_interpolation(t, start, end):
    return start + (end - start) * t


def identity_or_duplicated_value(possible_tuple):
    if isinstance(possible_tuple, tuple):
        return possible_tuple
    else:
        return possible_tuple, possible_tuple


class SoftHardSimulation(ModelLearningSimulation):
    def __init__(self, name, max_samples, greed, step_size, discount_rate,
                 gamma_optimistic, gamma_hard, lambda_hard, gamma_soft,
                 q_x_seed, q_y_seed, s_x_seed, s_y_seed, dataset_type, dataset_params,
                 shape, every, glie_start):
        dynamics_parameters = {
            'shape': shape
        }
        self.env = LowGoalSlip(dynamics_parameters=dynamics_parameters)

        self.q_hyperparameters = {
            'outputscale_prior': (0.4, 2),
            'lengthscale_prior': (0.05, 0.1),
            'noise_prior': (0.001, 0.002),
            'dataset_type': dataset_type,
            'dataset_params': dataset_params,
        }
        self.s_hyperparameters = {
            'outputscale_prior': (0.4, 2),
            'lengthscale_prior': (0.2, 0.1),
            'noise_prior': (0.001, 0.002),
            'dataset_type': dataset_type,
            'dataset_params': dataset_params,
        }
        self.q_x_seed = q_x_seed
        self.q_y_seed = q_y_seed
        self.s_x_seed = s_x_seed
        self.s_y_seed = s_y_seed

        self.gamma_optimistic_start, self.gamma_optimistic_end = identity_or_duplicated_value(gamma_optimistic)
        self.gamma_hard_start, self.gamma_hard_end = identity_or_duplicated_value(gamma_hard)
        self.lambda_hard_start, self.lambda_hard_end = identity_or_duplicated_value(lambda_hard)
        self.gamma_soft_start, self.gamma_soft_end = identity_or_duplicated_value(gamma_soft)
        self.gamma_optimistic = self.gamma_optimistic_start
        self.gamma_hard = self.gamma_hard_start
        self.gamma_soft = self.gamma_soft_start
        self.lambda_hard = self.lambda_hard_start

        self.agent = SoftHardLearner(
            self.env,
            greed=greed,
            step_size=step_size,
            discount_rate=discount_rate,
            q_x_seed=self.q_x_seed,
            q_y_seed=self.q_y_seed,
            gamma_optimistic=self.gamma_optimistic,
            gamma_hard=self.gamma_hard,
            lambda_hard=self.lambda_hard,
            gamma_soft=self.gamma_soft,
            s_x_seed=s_x_seed,
            s_y_seed=s_y_seed,
            q_gp_params=self.q_hyperparameters,
            s_gp_params=self.s_hyperparameters,
        )

        self.ground_truth = SafetyTruth(self.env)
        self.ground_truth.from_vibly_file(
            Path(__file__).parent.parent.parent / 'data' / 'ground_truth' /
            'from_vibly' / 'slip_map.pickle'
        )

        plotters = {
            'Q-Values_Safety': SoftHardPlotter(self.agent, self.ground_truth, ensure_in_dataset=True)
        }

        output_directory = Path(__file__).parent.resolve()
        super(SoftHardSimulation, self).__init__(output_directory, name,
                                                 plotters)

        self.max_samples = max_samples
        self.every = every
        if isinstance(glie_start, float):
            self.glie_start = int(glie_start * self.max_samples)
        else:
            self.glie_start = glie_start

    def get_models_to_save(self):
        # The keys must be the same as the actual names of the attributes, this is used in load_models.
        # This is hacky and should be replaced
        return {
            'Q_model': self.agent.Q_model,
            'safety_model': self.agent.safety_model
        }

    def load_models(self, skip_local=False):
        from edge.model.safety_models import MaternSafety
        from edge.model.value_models import GPQLearning
        models_names = list(self.get_models_to_save().keys())
        loaders= {
            'Q_model': lambda mpath: GPQLearning(mpath, self.env, self.q_x_seed, self.q_y_seed),
            'safety_model': lambda mpath: MaternSafety(mpath, self.env, self.gamma_optimistic,
                                                       self.s_x_seed, self.s_y_seed),
        }
        for mname in models_names:
            if not skip_local:
                load_path = self.local_models_path / mname
            else:
                load_path = self.models_path / mname
            setattr(
                self.agent,
                mname,
                loaders[mname](load_path)
            )

    def run(self):
        n_samples = 0
        self.save_figs(prefix='0')

        # train hyperparameters
        print('Optimizing hyperparameters...')
        s_train_x, s_train_y = self.ground_truth.get_training_examples()
        self.agent.fit_models(
            s_epochs=50, s_train_x=s_train_x, s_train_y=s_train_y, s_optimizer_kwargs={'lr': 0.1}
        )
        self.agent.fit_models(
            s_epochs=50, s_train_x=s_train_x, s_train_y=s_train_y, s_optimizer_kwargs={'lr': 0.01}
        )
        self.agent.fit_models(
            s_epochs=50, s_train_x=s_train_x, s_train_y=s_train_y, s_optimizer_kwargs={'lr': 0.001}
        )
        print('Lengthscale:',self.agent.safety_model.gp.covar_module.base_kernel.lengthscale)
        print('Outputscale:',self.agent.safety_model.gp.covar_module.outputscale)
        print('Done.')
        print('Training...')
        while n_samples < self.max_samples:
            reset_state = self.agent.get_random_safe_state()
            self.agent.reset(reset_state)
            failed = self.agent.failed
            n_steps = 0
            while not failed and n_steps < 50:
                n_samples += 1
                n_steps += 1
                old_state = self.agent.state
                new_state, reward, failed = self.agent.step()
                action = self.agent.last_action

                # * start reducing eps to converge to a greedy policy.
                if self.glie_start is not None and n_samples > self.glie_start:
                    self.agent.greed *= (n_samples - self.glie_start) / (
                                        (n_samples - self.glie_start + 1))
                self.agent.gamma_optimistic = affine_interpolation(
                    n_samples / self.max_samples,
                    self.gamma_optimistic_start,
                    self.gamma_optimistic_end
                )
                self.agent.gamma_hard = affine_interpolation(
                    n_samples / self.max_samples,
                    self.gamma_hard_start,
                    self.gamma_hard_end
                )
                self.agent.lambda_hard = affine_interpolation(
                    n_samples / self.max_samples,
                    self.lambda_hard_start,
                    self.lambda_hard_end
                )
                self.agent.gamma_soft = affine_interpolation(
                    n_samples / self.max_samples,
                    self.gamma_soft_start,
                    self.gamma_soft_end
                )

                color = None if not self.agent.updated_safety else [0.3, 0.3, 0.9]
                self.on_run_iteration(n_samples, old_state, action, new_state,
                                      reward, failed, color=color)

                if n_samples >= self.max_samples:
                    break
            self.agent.reset()
        print('Done.')

        self.save_figs(prefix=f'{self.name}_final')
        self.compile_gif()

    def on_run_iteration(self, n_samples, *args, **kwargs):
        super(SoftHardSimulation, self).on_run_iteration(*args, **kwargs)

        print(f'Iteration {n_samples}/{self.max_samples}')
        if n_samples % self.every == 0:
            self.save_figs(prefix=f'{n_samples}')


if __name__ == '__main__':
    sim = SoftHardSimulation(
        name='timeforgetting_test',
        max_samples=1000,
        greed=0.1,
        step_size=0.6,
        discount_rate=0.9,
        gamma_optimistic=(0.7, 0.9),
        gamma_hard=(0.71, 0.9),
        lambda_hard=(0, 0.05),
        gamma_soft=(0.8, 0.95),
        q_x_seed=np.array([0.4, 0.6]),
        q_y_seed=np.array([1]),
        s_x_seed=np.array([[0.4, 0.6], [0.8, 0.4]]),
        s_y_seed=np.array([1, 0.8]),
        dataset_type='timeforgetting',
        dataset_params={'keep': 200},
        shape=(201,201),
        every=100,
        glie_start=0.9
    )
    sim.set_seed(0)

    sim.run()
    sim.save_models()