from pathlib import Path
import numpy as np
import logging

from edge import ModelLearningSimulation
from edge.graphics.plotter import SoftHardPlotter, RewardFailurePlotter
from edge.model.safety_models import SafetyTruth
from edge.utils.logging import config_msg

from soft_hard_parameterization import LowGoalSlip, LowGoalHovership, \
    CartPole, LunarLander, SoftHardLearner


def affine_interpolation(t, start, end):
    return start + (end - start) * t


def identity_or_duplicated_value(possible_tuple):
    if isinstance(possible_tuple, tuple):
        return possible_tuple
    else:
        return possible_tuple, possible_tuple


class SoftHardSimulation(ModelLearningSimulation):
    def __init__(self, name, env_name, reward_threshold, max_samples, max_steps,
                 greed, step_size, discount_rate, gamma_optimistic, gamma_hard,
                 lambda_hard, gamma_soft, q_x_seed, q_y_seed, s_x_seed,
                 s_y_seed, optimize_hyperparameters, dataset_type,
                 dataset_params, shape, every, glie_start, reset_in_safe_state):
        parameterization = {
            'env_name':env_name,
            'max_samples':max_samples,
            'greed':greed,
            'step_size':step_size,
            'discount_rate':discount_rate,
            'gamma_optimistic':gamma_optimistic,
            'gamma_hard':gamma_hard,
            'lambda_hard':lambda_hard,
            'gamma_soft':gamma_soft,
            'q_x_seed':q_x_seed,
            'q_y_seed':q_y_seed,
            's_x_seed':s_x_seed,
            's_y_seed':s_y_seed,
            'optimize_hyperparameters':optimize_hyperparameters,
            'dataset_type':dataset_type,
            'dataset_params':dataset_params,
            'shape':shape,
            'every':every,
            'glie_start':glie_start,
            'reset_in_safe_state':reset_in_safe_state
        }
        dynamics_parameters = {
            'shape': shape
        }
        if env_name == 'slip':
            self.env = LowGoalSlip(dynamics_parameters=dynamics_parameters,
                                   reward_done_threshold=reward_threshold)
        elif env_name == 'hovership':
            self.env = LowGoalHovership(dynamics_parameters=dynamics_parameters,
                                        reward_done_threshold=reward_threshold)
        elif env_name == 'cartpole':
            self.env = CartPole(discretization_shape=shape)
        elif env_name == 'lander':
            self.env = LunarLander(discretization_shape=shape)

        self.q_hyperparameters = {
            'outputscale_prior': (0.12, 0.01),
            'lengthscale_prior': (0.15, 0.05),
            'noise_prior': (0.001, 0.002),
            'dataset_type': dataset_type,
            'dataset_params': dataset_params,
        }
        self.s_hyperparameters = {
            'outputscale_prior': (0.12, 0.01),
            'lengthscale_prior': (0.15, 0.05),
            'noise_prior': (0.001, 0.002),
            'dataset_type': dataset_type,
            'dataset_params': dataset_params,
        }
        self.q_x_seed = q_x_seed
        self.q_y_seed = q_y_seed
        self.s_x_seed = s_x_seed
        self.s_y_seed = s_y_seed
        self.optimize_hyperparameters = optimize_hyperparameters

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

        if env_name == 'slip':
            truth_path = Path(__file__).parent.parent.parent / 'data' / \
                         'ground_truth' / 'from_vibly' / 'slip_map.pickle'
        elif env_name == 'hovership':
            truth_path = Path(__file__).parent.parent.parent / 'data' / \
                         'ground_truth' / 'from_vibly' / 'hover_map.pickle'
        else:
            truth_path = None
        if truth_path is not None:
            self.ground_truth = SafetyTruth(self.env)
            self.ground_truth.from_vibly_file(truth_path)
        else:
            self.ground_truth = None

        plottable_Q = ['slip', 'hovership']
        if env_name in plottable_Q:
            plotters = {
                'Q-Values_Safety': SoftHardPlotter(self.agent, self.ground_truth, ensure_in_dataset=True)
            }
        else:
            plotters = {}
        plotters.update({
            'RewardFailure': RewardFailurePlotter(
                agents_names=['Soft-hard'],
                window_size=10,
                padding_value=1
            )
        })

        output_directory = Path(__file__).parent.resolve()
        super(SoftHardSimulation, self).__init__(output_directory, name,
                                                 plotters)

        self.max_samples = max_samples
        self.max_steps = max_steps
        self.every = every
        if isinstance(glie_start, float):
            self.glie_start = int(glie_start * self.max_samples)
        else:
            self.glie_start = glie_start
        self.reset_in_safe_state = reset_in_safe_state

        msg = ''
        for pname, pval in parameterization.items():
            msg += pname + ' = ' + str(pval) + ', '
        msg = msg[:-2]
        logging.info(config_msg(f'Simulation started with parameters: {msg}'))

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
        CV_SETS = 1000
        n_samples = 0
        self.save_figs(prefix='0')

        if self.optimize_hyperparameters:
            logging.info('Optimizing hyperparameters...')
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
            logging.info('Done.')
        else:
            logging.info('Hyperparameters were NOT optimized.')
        logging.info(config_msg(
            'Lengthscale:'
            f'{self.agent.safety_model.gp.covar_module.base_kernel.lengthscale}'
        ))
        logging.info(config_msg(
            'Outputscale:'
            f'{self.agent.safety_model.gp.covar_module.outputscale}'
        ))
        logging.info('Training...')
        while n_samples < self.max_samples:
            if self.reset_in_safe_state:
                reset_state = self.agent.get_random_safe_state()
            else:
                reset_state = None
            self.agent.reset(reset_state)
            failed = self.agent.failed
            done = self.env.done
            n_steps = 0
            while not done and n_steps < self.max_steps:
                n_samples += 1
                n_steps += 1
                old_state = self.agent.state
                new_state, reward, failed, done = self.agent.step()
                action = self.agent.last_action

                # * start reducing step size so Q-Learning converges
                if self.glie_start is not None and n_samples > self.glie_start:
                    self.agent.step_size *= (n_samples - self.glie_start) / (
                                        (n_samples - self.glie_start + 1))
                self.agent.gamma_optimistic = affine_interpolation(
                    min(n_samples / CV_SETS, 1),
                    self.gamma_optimistic_start,
                    self.gamma_optimistic_end
                )
                self.agent.gamma_hard = affine_interpolation(
                    min(n_samples / CV_SETS, 1),  # n_samples / self.max_samples,
                    self.gamma_hard_start,
                    self.gamma_hard_end
                )
                self.agent.lambda_hard = affine_interpolation(
                    min(n_samples / CV_SETS, 1),  # n_samples / self.max_samples,
                    self.lambda_hard_start,
                    self.lambda_hard_end
                )
                self.agent.gamma_soft = affine_interpolation(
                    min(n_samples / CV_SETS, 1),  # n_samples / self.max_samples,
                    self.gamma_soft_start,
                    self.gamma_soft_end
                )

                color = None if not self.agent.updated_safety else [0.3, 0.3, 0.9]
                self.on_run_iteration(
                    n_samples=n_samples,
                    state=old_state,
                    action=action,
                    new_state=new_state,
                    reward=reward,
                    failed=failed,
                    done=done,
                    color=color,
                    aname='Soft-hard'
                )

                if n_samples >= self.max_samples:
                    break
        logging.info('Done.')

        self.save_figs(prefix=f'{self.name}_final')
        self.compile_gif()

    def on_run_iteration(self, n_samples, *args, **kwargs):
        super(SoftHardSimulation, self).on_run_iteration(*args, **kwargs)

        logging.info(f'Iteration {n_samples}/{self.max_samples}')
        logging.info(f'# of Q-values training examples: '
              f'{len(self.agent.Q_model.gp.train_x)}')
        logging.info(f'# of safety measure training examples: '
              f'{len(self.agent.safety_model.gp.train_x)}')
        if kwargs['failed']:
            logging.info('Failed!')
        elif kwargs['done']:
            logging.info('Solved!')
        if kwargs['failed'] and n_samples > 900:
            print('pause')
        if n_samples % self.every == 0:
            self.save_figs(prefix=f'{n_samples}')

        self.env.render()


if __name__ == '__main__':
    import time

    env_dependent_params = {
        'slip': {
            'q_x_seed': np.array([0.4, 0.6]),
            'q_y_seed': np.array([1]),
            's_x_seed': np.array([[0.4, 0.6], [0.8, 0.4]]),
            's_y_seed': np.array([1, 0.8]),
            'shape': (101, 101),
            'reward_threshold': 10,
        },
        'hovership': {
            'q_x_seed': np.array([[1.3, 0.6], [2, 0]]),
            'q_y_seed': np.array([1, 1]),
            's_x_seed': np.array([[1.3, 0.6], [1.8, 0.2]]),
            's_y_seed': np.array([1, 1]),
            'shape': (101, 101),
            'reward_threshold': 100,
        },
        'cartpole': {
            'q_x_seed': np.array([[0, 0, 0, 0, 0]]),
            'q_y_seed': np.array([200]),
            's_x_seed': np.array([
                                    [0, 0,    0, 0, 0],
                                    [0, 0, -0.4, 0, 0],
                                    [0, 0,  0.4, 0, 0]
                                 ]),
            's_y_seed': np.array([10, 0.1, 0.1]),
            'shape': (10, 10, 10, 10, 10),
            'reward_threshold': None,
        },
        # State is: [x, y, vx, vy, theta, omega, left_contact, right_contact]
        # Action is: [main, left_right]
        # Number of dims: 8 + 2
        'lander': {
            'q_x_seed': np.array([
                [0,   0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1.4, 0, 0, 0, 0, 0, 0, 1, 0]
            ]),
            'q_y_seed': np.array([200, 100]),
            's_x_seed': np.array([
                [0, 1.4, 0, 0, 0, 0, 0, 0, 1, 0]
            ]),
            's_y_seed': np.array([1]),
            'shape': (10, 10, 10, 10, 10, 10, 10, 10, 10, 10),
            'reward_threshold': None,
        },
    }

    ENV_NAME = 'hovership'
    sim = SoftHardSimulation(
        name=f'{ENV_NAME}_failure_exploration',
        env_name=ENV_NAME,
        reward_threshold=env_dependent_params[ENV_NAME]['reward_threshold'],
        max_samples=1000,
        max_steps=np.inf,
        greed=0.1,
        step_size=0.6,
        discount_rate=0.9,
        gamma_optimistic=(0.6, 0.9),
        gamma_hard=(0.61, 0.9),
        lambda_hard=(0, 0),
        gamma_soft=(0.7, 0.95),
        q_x_seed=env_dependent_params[ENV_NAME]['q_x_seed'],
        q_y_seed=env_dependent_params[ENV_NAME]['q_y_seed'],
        s_x_seed=env_dependent_params[ENV_NAME]['s_x_seed'],
        s_y_seed=env_dependent_params[ENV_NAME]['s_y_seed'],
        optimize_hyperparameters=False,
        dataset_type='neighborerasing',
        dataset_params={'radius': 0.01},  # {'keep': 200},  # {'radius': 0.01},
        shape=env_dependent_params[ENV_NAME]['shape'],
        every=100,
        glie_start=0.9,
        reset_in_safe_state=True  # True is expensive, and useless for Gym
    )
    sim.set_seed(value=0)

    t0 = time.time()
    sim.run()
    t1 = time.time()
    dt = t1 - t0
    logging.info(f'Simulation duration: {dt:.2f} s')
    sim.save_models()