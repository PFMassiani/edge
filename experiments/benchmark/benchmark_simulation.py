import logging
import numpy as np
from pathlib import Path
from edge.utils.logging import config_msg
from edge import ModelLearningSimulation
from edge.model.safety_models import SafetyTruth
from edge.agent import QLearner
from benchmark_environments import LowGoalSlip, PenalizedSlip, \
    LowGoalHovership, PenalizedHovership
from benchmark_agents import SoftHardLearner, EpsilonSafety, \
    SafetyQLearningSwitcher
from edge.graphics.plotter import QValueAndSafetyPlotter, QValuePlotter

logger = logging.getLogger(__name__)

ENV_CONSTRUCTOR = {
    'low_goal_slip': LowGoalSlip,
    'penalized_slip': PenalizedSlip,
    'low_goal_hovership': LowGoalHovership,
    'penalized_hovership': PenalizedHovership,
}
AGENT_CONSTRUCTOR = {
    'q_learner': QLearner,
    'soft_hard_learner': SoftHardLearner,
    'epsilon_safety': EpsilonSafety,
    'switcher': SafetyQLearningSwitcher,
}
VIBLY_DATA_PATH = Path('../../data/ground_truth/from_vibly')
SAFETY_TRUTH_PATH = {
    'low_goal_slip': VIBLY_DATA_PATH / 'slip_map.pickle',
    'penalized_slip': VIBLY_DATA_PATH / 'slip_map.pickle',
    'low_goal_hovership': VIBLY_DATA_PATH / 'hover_map.pickle',
    'penalized_hovership': VIBLY_DATA_PATH / 'hover_map.pickle',
}
SAFETY_TRUTH_FROM_VIBLY = ['low_goal_slip', 'penalized_slip',
                           'low_goal_hovership', 'penalized_hovership']
PLOTTABLE_Q = ['low_goal_slip', 'penalized_slip',
               'low_goal_hovership', 'penalized_hovership']
HAS_SAFETY_MODEL = ['soft_hard_learner', 'epsilon_safety', 'switcher']

FAILURE_SAMPLE_COLOR = [0.3, 0.3, 0.9]  # should be in edge.graphics


class AgentMetrics:
    def __init__(self, *metrics_names):
        self.metrics = {mname: [] for mname in metrics_names}

    def add_measurement(self, index, *args):
        for mname, mval in args:
            self.metrics[mname].append((index, mval))

    def save(self, filename):
        filename = str(filename)
        save_dict = {mname: np.array(mvalues)
                     for mname, mvalues in self.metrics.items()}
        np.savez(filename, **save_dict)

    @staticmethod
    def load(filename):
        filename = str(filename)
        loaded_dict = np.load(filename, allow_pickle=False)
        self = AgentMetrics(list(loaded_dict.keys()))
        self.metrics = loaded_dict
        return self


class BenchmarkSingleSimulation(ModelLearningSimulation):
    EXP_REWARD_MNAME = 'expected_reward'
    EXP_FAILURE_MNAME = 'expected_failure'
    STD_REWARD_MNAME = 'std_reward'
    STD_FAILURE_MNAME = 'std_failure'
    Q_V_Q_C_MNAME = 'Q_V_minus_Q_cautious'
    Q_C_Q_V_MNAME = 'Q_cautious_minus_Q_V'
    METRICS_BASE_NAMES = [EXP_REWARD_MNAME, EXP_FAILURE_MNAME,
                          STD_REWARD_MNAME, STD_FAILURE_MNAME]

    def __init__(self, output_directory, name,
                 envname, aname, envparams, aparams,
                 n_episodes, glie_start, safety_parameters_update_end,
                 reset_in_safe_state, metrics_sampling_frequency,
                 n_episodes_in_measurement, plot_every):
        self.env = ENV_CONSTRUCTOR[envname](**envparams)
        self.agent = AGENT_CONSTRUCTOR[aname](env=self.env, **aparams)
        safety_truth_path = SAFETY_TRUTH_PATH[envname]
        if envname in SAFETY_TRUTH_FROM_VIBLY:
            self.safety_truth = SafetyTruth(self.env)
            self.safety_truth.from_vibly_file(safety_truth_path)
        else:
            self.safety_truth = SafetyTruth.load(safety_truth_path, self.env)

        self.n_episodes = n_episodes
        self.glie_start = glie_start if not isinstance(glie_start, float) else \
            int(glie_start * self.n_episodes)
        if safety_parameters_update_end is not None:
            if isinstance(safety_parameters_update_end, float):
                update_end = int(safety_parameters_update_end * n_episodes)
                self.safety_parameters_update_end = update_end
            else:
                self.safety_parameters_update_end = safety_parameters_update_end
        else:
            self.safety_parameters_update_end = n_episodes
        self.reset_in_safe_state = reset_in_safe_state
        self.metrics_sampling_frequency = metrics_sampling_frequency
        self.n_episodes_in_measurement = n_episodes_in_measurement
        self.plot_every = plot_every
        self.agent_has_safety_model = aname in HAS_SAFETY_MODEL

        self.METRICS_NAMES = BenchmarkSingleSimulation.METRICS_BASE_NAMES
        if self.agent_has_safety_model:
            self.METRICS_NAMES += [BenchmarkSingleSimulation.Q_C_Q_V_MNAME,
                                   BenchmarkSingleSimulation.Q_V_Q_C_MNAME]

        plotters = {}
        if envname in PLOTTABLE_Q:
            if self.agent_has_safety_model:
                plotters.update({
                    'Q-Values_Safety': QValueAndSafetyPlotter(
                        self.agent,
                        self.safety_truth,
                        # ensure_in_dataset=True
                    )
                })
            else:
                plotters.update({
                    'Q-Values': QValuePlotter(
                        self.agent,
                        self.safety_truth,
                        write_values=False,
                        plot_samples=True,
                    )
                })

        super(BenchmarkSingleSimulation, self).__init__(output_directory, name,
                                                        plotters)

        self.metrics_path = self.output_directory / 'metrics'
        self.metrics = AgentMetrics(*self.METRICS_NAMES)

        simparams = {
            'output_directory': output_directory,
            'name': name,
            'n_episodes': n_episodes,
            'glie_start': glie_start,
            'safety_parameters_update_end': safety_parameters_update_end,
            'reset_in_safe_state': reset_in_safe_state,
            'metrics_sampling_frequency': metrics_sampling_frequency,
            'n_episodes_in_measurement': n_episodes_in_measurement,
            'plot_every': plot_every,
        }
        logger.info(config_msg(f"Setting up simulation {name}"))
        logger.info(config_msg(f"ENVIRONMENT: {envname}"))
        logger.info(config_msg(str(envparams)))
        logger.info(config_msg(f"AGENT: {aname}"))
        logger.info(config_msg(str(aparams)))
        logger.info(config_msg("SIMULATION:"))
        logger.info(config_msg(str(simparams)))

    def get_models_to_save(self):
        if self.agent_has_safety_model:
            return {
                'Q_model': self.agent.Q_model,
                'safety_model': self.agent.safety_model
            }
        else:
            return {
                'Q_model': self.agent.Q_model,
            }

    def load_models(self, skip_local=False):
        pass

    def get_random_safe_state(self):
        viable_state_indexes = np.argwhere(self.safety_truth.viability_kernel)
        chosen_index_among_safe = np.random.choice(
            viable_state_indexes.shape[0]
        )
        chosen_index = tuple(viable_state_indexes[chosen_index_among_safe])
        safe_state = self.env.state_space[chosen_index]

        return safe_state

    def on_run_episode_iteration(self, *args, **kwargs):
        super(BenchmarkSingleSimulation, self).on_run_iteration(*args, **kwargs)

    def on_run_iteration(self, n_ep):
        if n_ep % self.plot_every == 0:
            self.save_figs(prefix=f'{n_ep}')

    def run_episode(self):
        episode = []
        reset_state = None if not self.reset_in_safe_state else \
            self.get_random_safe_state()
        # We don't allow initializing in failure directly, even when
        # reset_in_safe_state == False
        done = True
        while done:
            self.agent.reset(reset_state)
            done = self.env.done
        while not done:
            old_state = self.agent.state
            new_state, reward, failed = self.agent.step()
            done = self.env.done
            action = self.agent.last_action
            episode.append((old_state, action, new_state, reward, failed, done))
            if self.agent.training_mode:
                if self.agent_has_safety_model:
                    color = None if not self.agent.updated_safety else \
                        FAILURE_SAMPLE_COLOR
                else:
                    color = None
                self.on_run_episode_iteration(
                    state=old_state,
                    action=action,
                    new_state=new_state,
                    reward=reward,
                    failed=failed,
                    done=done,
                    color=color,
                )
        return episode

    def run(self):
        training_episodes = [None] * self.n_episodes
        for n_ep in range(self.n_episodes):
            self.agent.training_mode = True
            episode = self.run_episode()
            training_episodes[n_ep] = episode

            try:
                total_reward = sum(list(zip(*episode))[3])
                failed = 'failed' if episode[-1][4] else 'success'
            except IndexError:
                total_reward = 0
                failed = 'failed'
            logging.info(f'Episode {n_ep}: {total_reward} reward | {failed}')
            msg = '\n'.join([str(epstep) for epstep in episode])
            logging.info(msg)

            if (n_ep >= 0) and (n_ep % self.metrics_sampling_frequency == 0):
                self.agent.training_mode = False
                measurement_episodes = [None] * self.n_episodes_in_measurement
                for n_measurement_ep in range(self.n_episodes_in_measurement):
                    measurement_episodes[n_measurement_ep] = self.run_episode()
                metrics_list = self.get_metrics(measurement_episodes)
                self.metrics.add_measurement(n_ep, *metrics_list)

            self.on_run_iteration(n_ep)

            if n_ep >= self.glie_start:
                self.agent.decrease_step_size()
            if self.agent_has_safety_model and \
                    (n_ep <= self.safety_parameters_update_end):
                t = (n_ep + 1) / self.safety_parameters_update_end
                self.agent.safety_parameters_affine_update(t)
            if isinstance(self.agent, SafetyQLearningSwitcher) and \
                    (n_ep == self.safety_parameters_update_end):
                self.agent.explore_safety = False

        self.metrics.save(self.metrics_path)

    def get_metrics(self, measurement_episodes):
        # measurement_episodes = np.array(measurement_episodes, dtype=float)
        episodes_lists = [list(zip(*ep)) for ep in measurement_episodes]
        rewards = [sum(ep_list[3]) for ep_list in episodes_lists]
        failures = [any(ep_list[4]) for ep_list in episodes_lists]
        # Metrics from measurements episodes
        exp_reward_metric = np.mean(rewards)
        std_reward_metric = np.std(rewards)
        exp_failure_metric = np.mean(failures)
        std_failure_metric = np.std(failures)
        metrics_values = [
            exp_reward_metric,
            exp_failure_metric,
            std_reward_metric,
            std_failure_metric,
        ]

        # Metrics that don't require measurement episodes
        if self.agent_has_safety_model:
            Q_cautious = self.agent.safety_model.level_set(
                state=None,  # Whole state-space
                lambda_threshold=self.agent.lambda_cautious,
                gamma_threshold=self.agent.gamma_cautious
            ).astype(int)
            Q_V = self.safety_truth.viable_set_like(
                self.env.stateaction_space
            ).astype(int)

            Q_cautious_Q_V = (Q_cautious - Q_V).clip(0, 1)
            Q_V_Q_cautious = (Q_V - Q_cautious).clip(0, 1)
            # The measure of the underlying sets is the mean value of each of
            # these arrays
            Q_cautious_Q_V_metric = Q_cautious_Q_V.sum() / Q_V.sum()
            Q_V_Q_cautious_metric = Q_V_Q_cautious.sum() / Q_V.sum()
            metrics_values += [
                Q_V_Q_cautious_metric,
                Q_cautious_Q_V_metric,
            ]

        return list(zip(self.METRICS_NAMES, metrics_values))


if __name__ == '__main__':
    envparams = {
            'penalty_level': 100,
            'dynamics_parameters': {'shape': (201, 201)},
            'reward_done_threshold': 50,
    }
    gp_params = {
        'outputscale_prior': (0.12, 0.01),
        'lengthscale_prior': (0.15, 0.05),
        'noise_prior': (0.001, 0.002),
        'dataset_type': 'neighborerasing',
        'dataset_params': {'radius': 0.1},
    }
    aparams_0 = {
        'greed': 0.1,
        'step_size': 0.6,
        'discount_rate': 0.9,
        'x_seed': np.array([[1.3, 0.6]]),
        'y_seed': np.array([1]),
        'gp_params': gp_params,
        'keep_seed_in_data': True,
    }
    aparams_1 = {
        'greed': 0.1,
        'step_size': 0.6,
        'discount_rate': 0.9,
        'q_x_seed': np.array([[1.3, 0.6], [2, 0]]),
        'q_y_seed': np.array([1, 1]),
        'gamma_optimistic': (0.6, 0.6),
        'gamma_cautious': (0.61, 0.61),
        'lambda_cautious': (0., 0.),
        's_x_seed': np.array([[1.3, 0.6], [1.8, 0.2]]),
        's_y_seed': np.array([1, 1]),
        'q_gp_params': gp_params,
        's_gp_params': gp_params,
        'keep_seed_in_data': True
    }
    aparams_2 = {
        'greed': 0.1,
        'step_size': 0.6,
        'discount_rate': 0.9,
        'q_x_seed': np.array([[1.3, 0.6], [2, 0]]),
        'q_y_seed': np.array([1, 1]),
        'gamma_optimistic': (0.6, 0.6),
        'gamma_hard': (0.61, 0.61),
        'lambda_hard': (0., 0.),
        'gamma_soft': (0.7, 0.7),
        's_x_seed': np.array([[1.3, 0.6], [1.8, 0.2]]),
        's_y_seed': np.array([1, 1]),
        'q_gp_params': gp_params,
        's_gp_params': gp_params,
        'keep_seed_in_data': True
    }

    sim = BenchmarkSingleSimulation(
        output_directory=Path('.').absolute().resolve(),
        name='test_2',
        envname='penalized_hovership',
        aname='switcher',
        envparams=envparams,
        aparams=aparams_1,
        n_episodes=5,
        glie_start=0.9,
        safety_parameters_update_end=0.5,
        reset_in_safe_state=True,
        metrics_sampling_frequency=2,
        n_episodes_in_measurement=2,
        plot_every=1
    )
    sim.run()
