from pathlib import Path
import logging
import numpy as np

from edge.simulation import ModelLearningSimulation
from edge.model.safety_models import SafetyTruth
from edge.graphics.plotter import QValuePlotter, QValueAndSafetyPlotter
from edge.utils.logging import config_msg
from edge.dataset import Dataset

from gpsarsa_test_parameterization import \
    LowGoalHovership, PenalizedHovership, HighGoalHovership, \
    HighPenalizedHovership, SARSALearner


class GPSARSATestSimulation(ModelLearningSimulation):
    def __init__(self, name, goal, shape, reset_in_safe_state, penalty,
                 steps_done_threshold, max_episodes, xi, discount_rate,
                 q_x_seed, q_y_seed, use_safety_model, s_x_seed, s_y_seed,
                 gamma_cautious, lambda_cautious, gamma_optimistic,
                 plot_every, measure_every, episodes_per_measure):
        parameterization = {
            'name': name,
            'goal': goal,
            'shape': shape,
            'reset_in_safe_state': reset_in_safe_state,
            'steps_done_threshold': steps_done_threshold,
            'max_episodes': max_episodes,
            'xi': xi,
            'discount_rate': discount_rate,
            'q_x_seed': q_x_seed,
            'q_y_seed': q_y_seed,
            'use_safety_model': use_safety_model,
            's_x_seed': s_x_seed,
            's_y_seed': s_y_seed,
            'gamma_cautious': gamma_cautious,
            'lambda_cautious': lambda_cautious,
            'gamma_optimistic': gamma_optimistic,
            'plot_every': plot_every,
            'measure_every': measure_every,
            'episodes_per_measure': episodes_per_measure
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

        truth_path = Path(__file__).absolute().parent.parent.parent / 'data' / \
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

        self.max_episodes = max_episodes
        self.plot_every = plot_every
        self.measure_every = measure_every
        self.episodes_per_measure = episodes_per_measure
        self.reset_in_safe_state = reset_in_safe_state

        self.testing_dataset = Dataset(*Dataset.DEFAULT_COLUMNS,
                                       group_name='measurement')

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

    def get_random_safe_state(self):
        viable_state_indexes = np.argwhere(self.ground_truth.viability_kernel)
        chosen_index_among_safe = np.random.choice(
            viable_state_indexes.shape[0]
        )
        chosen_index = tuple(viable_state_indexes[chosen_index_among_safe])
        safe_state = self.env.state_space[chosen_index]

        return safe_state

    def _append_to_episode(self, episode, state, action, new_state, reward,
                           failed, done):
        episode[self.training_dataset.STATE].append(state)
        episode[self.training_dataset.ACTION].append(action)
        episode[self.training_dataset.NEW].append(new_state)
        episode[self.training_dataset.REWARD].append(reward)
        episode[self.training_dataset.FAILED].append(failed)
        episode[self.training_dataset.DONE].append(done)

    def run_episode(self):
        done = self.env.done
        episode = {cname: []
                   for cname in self.training_dataset.columns_wo_group}
        while not done:
            old_state = self.agent.state
            new_state, reward, failed, done = self.agent.step()
            action = self.agent.last_action
            self._append_to_episode(episode, old_state, action, new_state,
                                    reward, failed, done)
            if self.agent.training_mode:
                self.on_run_episode_iteration(
                    state=old_state,
                    action=action,
                    new_state=new_state,
                    reward=reward,
                    failed=failed,
                    done=done,
                    color=[0, 1, 0] if self.agent.update_safety_model else None,
                )
        return episode

    def measure(self, n_episode):
        self.agent.training_mode = False
        for n_meas_ep in range(self.episodes_per_measure):
            done = True
            while done:
                reset_state = None if not self.reset_in_safe_state else \
                    self.get_random_safe_state()
                self.agent.reset(reset_state)
                done = self.agent.env.done
            meas_ep = self.run_episode()
            len_meas_ep = len(meas_ep[list(meas_ep.keys())[0]])
            meas_ep[self.testing_dataset.EPISODE] = [n_meas_ep] * len_meas_ep
            self.testing_dataset.add_group(meas_ep, group_number=n_episode)
            # TODO add actual measurements
        self.agent.training_mode = True

    def run(self):
        n_episode = 0
        self.save_figs(prefix='0')
        while n_episode < self.max_episodes:
            done = True
            while done:
                self.agent.reset()
                done = self.agent.env.done
            n_episode += 1
            episode = self.run_episode()
            self.training_dataset.add_group(episode, group_number=n_episode)
            if n_episode % self.measure_every == 0:
                self.measure(n_episode)
            self.on_run_iteration(n_episode)
        self.on_simulation_end()

    def on_run_episode_iteration(self, *args, **kwargs):
        super(GPSARSATestSimulation, self).on_run_iteration(*args, **kwargs)

    def on_run_iteration(self, n_episode):
        train = self.training_dataset
        episode = train.df.loc[train.df.loc[:, train.group_name] == n_episode]
        failed = episode.loc[episode.index[-1], train.FAILED]
        reward = episode.loc[:, train.REWARD].sum()
        logging.info(f'Episode {n_episode}/{self.max_episodes}: '
                     f'{"failed" if failed else "success"} '
                     f'| reward: {reward:.3f}')
        if n_episode % self.plot_every == 0:
            self.save_figs(prefix=f'{n_episode}')

    def on_simulation_end(self, *args, **kwargs):
        super().on_simulation_end(*args, **kwargs)
        self.testing_dataset.save(self.data_path / 'testing_samples.csv')


if __name__ == '__main__':
    import time

    seed = int(time.time())

    sim = GPSARSATestSimulation(
        name=f'full_test_{seed}',
        goal='low',  # 'low' or 'high'
        shape=(201, 201),
        reset_in_safe_state=True,
        penalty=100,
        steps_done_threshold=30,
        max_episodes=200,
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
        plot_every=50,
        measure_every=10,
        episodes_per_measure=5,
    )

    sim.set_seed(value=seed)
    logging.info(config_msg(f'Random seed: {seed}'))

    t0 = time.time()
    sim.run()
    t1 = time.time()
    dt = t1 - t0
    logging.info(f'Simulation duration: {dt:.2f} s')
    sim.save_models()
