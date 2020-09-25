from pathlib import Path
import logging
import numpy as np

from edge.simulation import ModelLearningSimulation
from edge.model.safety_models import SafetyTruth
from edge.graphics.plotter import QValuePlotter, QValueAndSafetyPlotter
from edge.utils.logging import config_msg

from gpsarsa_test_parameterization import \
    LowGoalHovership, PenalizedHovership, HighGoalHovership, \
    HighPenalizedHovership, SARSALearner


class GPSARSATestSimulation(ModelLearningSimulation):
    def __init__(self, name, goal, shape, penalty, steps_done_threshold,
                 max_samples, xi, discount_rate, q_x_seed, q_y_seed,
                 use_safety_model, s_x_seed, s_y_seed, gamma_cautious,
                 lambda_cautious, gamma_optimistic,
                 every):
        parameterization = {
            'name': name,
            'goal': goal,
            'shape': shape,
            'steps_done_threshold': steps_done_threshold,
            'max_samples': max_samples,
            'xi': xi,
            'discount_rate': discount_rate,
            'q_x_seed': q_x_seed,
            'q_y_seed': q_y_seed,
            'use_safety_model':use_safety_model,
            's_x_seed': s_x_seed,
            's_y_seed': s_y_seed,
            'gamma_cautious': gamma_cautious,
            'lambda_cautious': lambda_cautious,
            'gamma_optimistic': gamma_optimistic,
            'every': every,
        }
        dynamics_parameters = {'shape': shape}
        if penalty is None:
            hovership = LowGoalHovership if goal == 'low' else HighGoalHovership
            self.env = hovership(
                dynamics_parameters=dynamics_parameters,
                steps_done_threshold=steps_done_threshold
            )
        else:
            hovership = PenalizedHovership if goal == 'low' \
                else HighPenalizedHovership
            self.env = hovership(
                penalty_level=penalty,
                dynamics_parameters=dynamics_parameters,
                steps_done_threshold=steps_done_threshold
            )

        self.x_seed = q_x_seed
        self.y_seed = q_y_seed
        self.q_gp_params = {
            'train_x': q_x_seed,
            'train_y': q_y_seed,
            'outputscale_prior': (0.12, 0.01),
            'lengthscale_prior': (0.15, 0.05),
            'noise_prior': (0.001, 0.002),
            'dataset_type': None,
            'dataset_params': None,
            'value_structure_discount_factor': discount_rate,
        }
        self.s_gp_params = None if not use_safety_model else {
            'train_x': s_x_seed,
            'train_y': s_y_seed,
            'outputscale_prior': (0.12, 0.01),
            'lengthscale_prior': (0.15, 0.05),
            'noise_prior': (0.001, 0.002),
            'dataset_type': None,
            'dataset_params': None,
            'value_structure_discount_factor': None,
        }
        self.agent = SARSALearner(
            env=self.env,
            xi=xi,
            keep_seed_in_data=True,
            q_gp_params=self.q_gp_params,
            s_gp_params=self.s_gp_params,
            gamma_cautious=gamma_cautious if use_safety_model else None,
            lambda_cautious=lambda_cautious if use_safety_model else None,
            gamma_optimistic=gamma_optimistic if use_safety_model else None,
        )

        truth_path = Path(__file__).parent.parent.parent / 'data' / \
                     'ground_truth' / 'from_vibly' / 'hover_map.pickle'
        self.ground_truth = SafetyTruth(self.env)
        self.ground_truth.from_vibly_file(truth_path)

        v_extreme = 10 * 1 / (1 - discount_rate)
        plotters = {
            'Q-Values': QValuePlotter(
                self.agent, self.ground_truth, plot_samples=True,
                vmin=-v_extreme, vmax=v_extreme, scale='lin'
            )
        } if not use_safety_model else {
            'Safety_Q-Values': QValueAndSafetyPlotter(
                self.agent, self.ground_truth,
                vmin=-v_extreme, vmax=v_extreme, scale='lin'
            )
        }

        output_directory = Path(__file__).parent.resolve()
        super(GPSARSATestSimulation, self).__init__(
            output_directory, name, plotters
        )

        self.max_samples = max_samples
        self.every = every

        msg = ''
        for pname, pval in parameterization.items():
            msg += pname + ' = ' + str(pval) + ', '
        msg = msg[:-2]
        logging.info(config_msg(f'Simulation started with parameters: {msg}'))

    def get_models_to_save(self):
        # The keys must be the same as the actual names of the attributes,
        # this is used in load_models. This is hacky and should be replaced
        if self.agent.has_safety_model:
            return {
                'Q_model': self.agent.Q_model,
                'safety_model': self.agent.safety_model
            }
        else:
            return {
                'Q_model': self.agent.Q_model
            }

    def load_models(self, skip_local=False):
        # TODO will fail as it is: add loading of safety_model
        from edge.model.value_models import MaternGPSARSA
        mname = 'Q_model'
        if not skip_local:
            load_path = self.local_models_path / mname
        else:
            load_path = self.models_path / mname
        setattr(
            self.agent,
            mname,
            MaternGPSARSA.load(load_path, self.env, self.x_seed, self.y_seed)
        )

    def run(self):
        n_samples = 0
        self.save_figs(prefix='0')
        while n_samples < self.max_samples:
            self.agent.reset()
            failed = self.agent.failed
            done = self.env.done
            while not done:
                n_samples += 1
                old_state = self.agent.state
                new_state, reward, failed, done = self.agent.step()
                action = self.agent.last_action
                self.on_run_iteration(
                    n_samples=n_samples,
                    state=old_state,
                    action=action,
                    new_state=new_state,
                    reward=reward,
                    failed=failed,
                    done=done,
                    color=[0, 1, 0] if self.agent.update_safety_model else None,
                )
                if n_samples >= self.max_samples:
                    break
        self.save_figs(prefix='final')

    def on_run_iteration(self, n_samples, *args, **kwargs):
        super(GPSARSATestSimulation, self).on_run_iteration(*args, **kwargs)
        if n_samples % self.every == 0:
            print(f'Iteration {n_samples}/{self.max_samples}')
            if n_samples > 0:
                self.save_figs(prefix=f'{n_samples}')


if __name__ == '__main__':
    import time

    sim = GPSARSATestSimulation(
        name='all_iter_plot',
        goal='high',  # 'low' or 'high'
        shape=(201, 201),
        penalty=100,
        steps_done_threshold=20,
        max_samples=50,
        xi=0.01,
        discount_rate=0.8,
        q_x_seed=np.array([[1.3, 0.6], [2, 0]]),
        q_y_seed=np.array([3.5, 0]),
        use_safety_model=False,
        s_x_seed=np.array([[1.3, 0.6], [2, 0.4]]),
        s_y_seed=np.array([1, 1]),
        gamma_cautious=0.75,
        lambda_cautious=0.05,
        gamma_optimistic=0.6,
        every=1,
    )
    sim.set_seed(value=0)

    t0 = time.time()
    sim.run()
    t1 = time.time()
    dt = t1 - t0
    print(f'Simulation duration: {dt:.2f} s')
    sim.save_models()
