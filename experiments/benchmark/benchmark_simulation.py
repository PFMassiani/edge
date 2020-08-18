import logging
import numpy as np
from pathlib import Path
from edge.utils.logging import config_msg
from edge import ModelLearningSimulation
from edge.model.safety_models import SafetyTruth
from edge.agent import QLearner
from .benchmark_environments import LowGoalSlip, PenalizedSlip, \
    LowGoalHovership, PenalizedHovership
from .benchmark_agents import SoftHardLearner, EpsilonSafety, \
    SafetyQLearningSwitcher

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
    METRICS_NAMES = [EXP_REWARD_MNAME, EXP_FAILURE_MNAME,
                     STD_REWARD_MNAME, STD_FAILURE_MNAME]

    def __init__(self, output_directory, name,
                 envname, aname, envparams, aparams,
                 n_episodes, glie_start, safety_parameters_update_end,
                 reset_in_safe_state, metrics_sampling_frequency,
                 n_episodes_in_measurement):
        self.env = ENV_CONSTRUCTOR[envname](**envparams)
        self.agent = AGENT_CONSTRUCTOR[aname](env=self.env, **aparams)
        safety_truth_path = SAFETY_TRUTH_PATH[envname]
        if envname in SAFETY_TRUTH_FROM_VIBLY:
            self.safety_truth = SafetyTruth(self.env)
            self.safety_truth.from_vibly_file(safety_truth_path)
        else:
            self.safety_truth = SafetyTruth.load(safety_truth_path, self.env)

        plotters = {}  # TODO

        super(BenchmarkSingleSimulation, self).__init__(output_directory, name,
                                                        plotters)

        self.metrics_path = self.output_directory / 'metrics'
        self.metrics = AgentMetrics(BenchmarkSingleSimulation.METRICS_NAMES)

        self.n_episodes = n_episodes
        self.glie_start = glie_start if not isinstance(glie_start, float) else \
            int(glie_start * self.n_episodes)
        self.safety_parameters_update_end = safety_parameters_update_end
        self.reset_in_safe_state = reset_in_safe_state
        self.metrics_sampling_frequency = metrics_sampling_frequency
        self.n_episodes_in_measurement = n_episodes_in_measurement

        simparams = {
            'output_directory': output_directory,
            'name': name,
            'n_episodes': n_episodes,
            'glie_start': glie_start,
            'safety_parameters_update_end': safety_parameters_update_end,
            'reset_in_safe_state': reset_in_safe_state,
            'metrics_sampling_frequency': metrics_sampling_frequency,
            'n_episodes_in_measurement': n_episodes_in_measurement
        }
        logger.info(config_msg(f"Setting up simulation {name}"))
        logger.info(config_msg(f"ENVIRONMENT: {envname}"))
        logger.info(config_msg(str(envparams)))
        logger.info(config_msg(f"AGENT: {aname}"))
        logger.info(config_msg(str(aparams)))
        logger.info(config_msg("SIMULATION:"))
        logger.info(config_msg(str(simparams)))

    def get_models_to_save(self):
        try:
            return {
                'Q_model': self.agent.Q_model,
                'safety_model': self.agent.safety_model
            }
        except AttributeError:
            return {
                'Q_model': self.agent.Q_model,
            }

    def load_models(self, skip_local=False):
        pass

    def run_episode(self):
        episode = []
        reset_state = self.agent.get_random_safe_state() \
            if self.reset_in_safe_state else None
        self.agent.reset(reset_state)
        done = self.env.done
        while not done:
            old_state = self.agent.state
            new_state, reward, failed, done = self.agent.step()
            action = self.agent.last_action
            episode.append((old_state, action, new_state, reward, failed, done))
        return episode

    def run(self):
        training_episodes = [] * self.n_episodes
        for n_ep in range(self.n_episodes):
            self.agent.training_mode = True
            training_episodes[n_ep] = self.run_episode()

            if n_ep % self.metrics_sampling_frequency == 0:
                self.agent.training_mode = False
                measurement_episodes = [] * self.n_episodes_in_measurement
                for n_measurement_ep in range(self.n_episodes_in_measurement):
                    measurement_episodes[n_measurement_ep] = self.run_episode()
                metrics_list = self.get_metrics_from_measurement_episodes(
                    measurement_episodes
                )
                self.metrics.add_measurement(index=n_ep, *metrics_list)

        self.metrics.save(self.metrics_path)

    def get_metrics_from_measurement_episodes(self, measurement_episodes):
        measurement_episodes = np.array(measurement_episodes, dtype=float)
        exp_reward_metric = measurement_episodes[:, 3].mean()
        std_reward_metric = measurement_episodes[:, 3].std()
        exp_failure_metric = measurement_episodes[:, 4].mean()
        std_failure_metric = measurement_episodes[:, 4].std()
        return list(zip(
            self.METRICS_NAMES,
            [
                exp_reward_metric,
                exp_failure_metric,
                std_reward_metric,
                std_failure_metric
            ]
        ))
