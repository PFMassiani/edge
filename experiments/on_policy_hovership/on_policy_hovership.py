from pathlib import Path
import logging
import time
import numpy as np
import torch

from edge.simulation import ModelLearningSimulation
from edge.graphics.plotter import SafetyPlotter
from edge.dataset import Dataset
from edge.utils.logging import config_msg
from edge.utils import device, cuda, timeit, log_simulation_parameters, \
    average_performances as general_perfs,\
    append_to_episode as general_append
from edge.model.safety_models import SafetyTruth

# noinspection PyUnresolvedReferences
from on_policy_agent import RandomSafetyLearner, AffineSafetyLearner
# noinspection PyUnresolvedReferences
from on_policy_environment import LowGoalHovership, LowGoalSlip


GROUP_NAME = 'Training'
SAFETY_NAME = 'Next safety'
CTRLR_VIAB = 'Controller is viable'
FLWD_CTRLR = 'Followed controller'


def append_to_episode(dataset, episode, state, action, new_state, reward,
                      failed, done, ctrlr_viab, flwd_ctrlr):
    general_append(dataset, episode, state, action, new_state, reward,
                   failed, done)
    episode[CTRLR_VIAB].append(ctrlr_viab)
    episode[FLWD_CTRLR].append(flwd_ctrlr)


def average_performances(df, group_name, episode_name, last_n_episodes=None):
    r, f = general_perfs(df, group_name, episode_name, last_n_episodes)
    exploration_steps = df[CTRLR_VIAB].astype(bool) & (
        ~(df[FLWD_CTRLR].astype(bool))
    )
    exploration_steps = exploration_steps.sum()
    off_controller_steps = (~df[FLWD_CTRLR].astype(bool)).sum()
    return r, f, exploration_steps, off_controller_steps

def cautious_qv(agent, safety_truth):
    Q_cautious = agent.safety_model.level_set(
        state=None,  # Whole state-space
        lambda_threshold=agent.lambda_cautious,
        gamma_threshold=agent.gamma_cautious
    ).astype(bool)
    Q_V = safety_truth.viable_set_like(
        agent.env.stateaction_space
    ).astype(bool)
    cautious_qv_ratio = (Q_V & Q_cautious).astype(int).sum()
    cautious_qv_ratio /= Q_V.astype(int).sum()
    return cautious_qv_ratio

def avg_reward_and_failure(df):
    r = df.groupby(Dataset.EPISODE)[Dataset.REWARD].sum().mean()
    f = df.groupby(Dataset.EPISODE)[Dataset.FAILED].any().mean()
    return r, f


class FixedControllerLowdim(ModelLearningSimulation):
    @log_simulation_parameters
    def __init__(self, name, shape,
                 gamma_cautious, lambda_cautious, gamma_optimistic,
                 controller, reset_in_safe_state,
                 n_episodes_train, n_episodes_test, n_train_test,
                 plot_every=1):
        shapedict = {} if shape is None else {'shape': shape}
        self.env = LowGoalHovership(
            goal_state=False,
            initial_state=np.array([1.3]),
            **shapedict  # This matters for the GP
        )

        x_seed = np.array([[2, .1]])
        y_seed = np.array([.5])
        lengthscale_means = (0.2, 0.2)
        lengthscale_vars = (0.1, 0.1)
        lengthscale_prior = tuple(zip(lengthscale_means, lengthscale_vars))
        outputscale_prior = (1., 10.)
        noise_prior = (0.007, 0.1)

        gp_params = {
            'train_x': x_seed,
            'train_y': y_seed,
            'outputscale_prior': outputscale_prior,
            'lengthscale_prior': lengthscale_prior,
            'noise_prior': noise_prior,
            'mean_constant': None,
            'dataset_type': None,
            'dataset_params': None,
            # Other possible options:
            # 'dataset_type': 'downsampling',
            # 'dataset_params': {'append_every': 10},
            # 'dataset_type': 'neighborerasing',
            # 'dataset_params': {'radius': 0.01},
            'value_structure_discount_factor': None,
        }
        if controller == 'random':
            agent = RandomSafetyLearner(
                env=self.env,
                s_gp_params=gp_params.copy(),
                gamma_cautious=gamma_cautious,
                lambda_cautious=lambda_cautious,
                gamma_optimistic=gamma_optimistic,
            )
        elif controller == 'affine':
            agent = AffineSafetyLearner(
                env=self.env,
                offset=(np.array([2.0]), np.array([0.1])),
                jacobian=np.array([[(0.7 - 0.1)/(0. - 2.)]]),
                s_gp_params=gp_params.copy(),
                gamma_cautious=gamma_cautious,
                lambda_cautious=lambda_cautious,
                gamma_optimistic=gamma_optimistic,
            )
        else:
            raise ValueError('Invalid controller')

        self.agent = agent

        truth_path = Path(__file__).parent.parent.parent / 'data' / \
                     'ground_truth' / 'from_vibly' / f'hover_map.pickle'
        self.ground_truth = SafetyTruth(self.env)
        self.ground_truth.from_vibly_file(truth_path)
        ctrlr = None if controller == 'random' else self.agent.policy
        plotters = {
            'safety': SafetyPlotter(
                self.agent, ground_truth=self.ground_truth, controller=ctrlr
            )
        }

        output_directory = Path(__file__).parent.resolve()
        super().__init__(output_directory, name, plotters)

        self.reset_in_safe_state = reset_in_safe_state
        self.n_episodes_train = n_episodes_train
        self.n_episodes_test = n_episodes_test
        self.n_train_test = n_train_test
        self.plot_every = plot_every

        self.training_dataset = Dataset(
            *Dataset.DEFAULT_COLUMNS, CTRLR_VIAB, FLWD_CTRLR,
            group_name=GROUP_NAME, name='train'
        )
        self.testing_dataset = Dataset(
                *Dataset.DEFAULT_COLUMNS, SAFETY_NAME, CTRLR_VIAB, FLWD_CTRLR,
                group_name=GROUP_NAME, name=f'test'
        )

    def run_episode(self, n_episode, prefix=None):
        episode = {cname: []
                   for cname in self.training_dataset.columns_wo_group}
        done = self.env.done
        n = 0
        if prefix is not None:
            self.save_figs(prefix=f'{prefix}_{n}')
        while not done:
            old_state = self.agent.state
            new_state, reward, failed, done = self.agent.step()
            action = self.agent.last_action
            ctrlr_action = self.agent.last_controller_action
            ctrlr_viab = self.ground_truth.is_viable(
                state=old_state, action=ctrlr_action
            )
            flwd_ctrlr = self.agent.followed_controller
            append_to_episode(self.training_dataset, episode, old_state, action,
                              new_state, reward, failed, done, ctrlr_viab,
                              flwd_ctrlr)
            if self.agent.training_mode:
                marker = None
                color = [1, 0, 0] if self.agent.followed_controller else [0, 1, 0]
                super().on_run_iteration(state=old_state, action=action,
                                         new_state=new_state, reward=reward,
                                         failed=failed, color=color,
                                         marker=marker)
                if prefix is not None:
                    if (n + 1) % self.plot_every == 0:
                        self.save_figs(prefix=f'{prefix}_{n}')
                n += 1
        len_episode = len(episode[self.training_dataset.REWARD])
        episode[self.training_dataset.EPISODE] = [n_episode] * len_episode
        return episode

    def reset_agent_state(self):
        if self.reset_in_safe_state:
            is_viable = self.agent.safety_model.measure(
                slice(None, None, None),
                lambda_threshold=self.agent.lambda_cautious,
                gamma_threshold=self.agent.gamma_cautious
            ) > 0
            if any(is_viable):
                viable_indexes = np.atleast_1d(np.argwhere(is_viable).squeeze())
                state_index = viable_indexes[np.random.choice(len(viable_indexes))]
                s = self.env.stateaction_space.state_space[state_index]
                self.agent.reset(s)
        while self.env.done:
            s = self.agent.reset()
        return s

    @timeit
    def train_agent(self, n_train):
        self.agent.training_mode = True
        # self.save_figs(prefix=f'{n_train}ep{0}')
        for n in range(self.n_episodes_train):
            self.reset_agent_state()
            episode = self.run_episode(n, prefix=f'{n_train}ep{n+1}')
            self.training_dataset.add_group(episode, group_number=n_train)
            # if (n+1) % self.plot_every == 0:
            #     self.save_figs(prefix=f'{n_train}ep{n+1}')

    @timeit
    def test_agent(self, n_test):
        self.agent.training_mode = False
        for n in range(self.n_episodes_test):
            self.reset_agent_state()
            episode = self.run_episode(n)
            self.testing_dataset.add_group(episode, group_number=n_test)

    @timeit
    def log_performance(self, n_train, ds, name_in_log, duration=None,
                        header=True, limit_episodes=None):
        df = ds.df
        if n_train is not None:
            train = df.loc[df[ds.group_name] == n_train, :]
        else:
            train = df
        r, f, xplo_steps, off_ctrlr = average_performances(
            train, ds.group_name, ds.EPISODE, limit_episodes
        )
        n_steps = len(train)
        caveat = '' if limit_episodes is None \
            else f'(last {limit_episodes} episodes) '
        header = '-------- Performance --------\n' if header else ''
        message = (f'--- {name_in_log} {caveat}\n'
                   f'Average total reward per episode: {r:.3f}\n'
                   f'Average number of failures: {f * 100:.3f} %\n'
                   f'Number of exploration steps: {xplo_steps} / {n_steps}\n'
                   f'Number of off-controller steps: {off_ctrlr} / {n_steps}')
        if duration is not None:
            message += f'\nComputation time: {duration:.3f} s'
        logging.info(header + message)

    def log_cautious_qv_ratio(self):
        ratio = cautious_qv(self.agent, self.ground_truth)
        message = f'Proportion of Q_V labeled as cautious: {ratio*100:.3f} %'
        logging.info(message)

    def log_memory(self):
        if device == cuda:
            message = ('Memory usage\n' + torch.cuda.memory_summary())
            logging.info(message)

    def log_samples(self):
        n_samples = self.agent.safety_model.gp.train_x.shape[0]
        logging.info(f'Training dataset size: {n_samples}')

    @timeit
    def checkpoint(self, n):
        self.training_dataset.save(self.data_path)
        self.testing_dataset.save(self.data_path)
        self.save_safety_model(f'safety_model_{n}')

    def save_safety_model(self, name):
        savepath = self.local_models_path / 'safety_model' / name
        savepath.mkdir(exist_ok=True, parents=True)
        self.agent.safety_model.save(savepath, save_data=True)

    def get_models_to_save(self):
        return {'safety_model': self.agent.safety_model}

    @timeit
    def run(self):
        for n in range(self.n_train_test):
            logging.info(f'========= CYCLE {n+1}/{self.n_train_test} ========')
            t = 0 if self.n_train_test == 1 else n / (self.n_train_test - 1)
            self.agent.update_safety_params(t=t)
            train_t = self.train_agent(n)
            try:
                pass
            except RuntimeError as e:
                train_t = None
                logging.critical(f'train_agent({n}) failed:\n{str(e)}')
                self.log_memory()
                torch.cuda.empty_cache()
            finally:
                self.log_performance(n, self.training_dataset, 'Training',
                                     train_t, header=True,
                                     limit_episodes=self.n_episodes_train)
            self.log_samples()
            try:
                test_t = self.test_agent(n)
            except RuntimeError as e:
                test_t = None
                logging.critical(f'test_agent({n}) failed:\n{str(e)}')
                torch.cuda.empty_cache()
            finally:
                self.log_performance(n, self.testing_dataset, 'Testing',
                                     test_t, header=False, limit_episodes=None)
            chkpt_t = self.checkpoint(n)
            logging.info(f'Checkpointing time: {chkpt_t:.3f} s')
        self.log_performance(None, self.training_dataset,
                             'Training - Full dataset', duration=None,
                             header=False, limit_episodes=None)
        self.log_performance(None, self.testing_dataset,
                             'Testing - Full dataset', duration=None,
                             header=False, limit_episodes=None)
        self.log_cautious_qv_ratio()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('controller', choices=['affine', 'random'], help='chosen controller')

    args = parser.parse_args()
    controller = args.controller
    if controller == 'affine':
        seed = 1605218995
    elif controller == 'random':
        seed = 1605262057
    else:
        raise ValueError('Unrecognized controller')

    name = f'{controller}_{seed}'
    sim = FixedControllerLowdim(
        name=name,
        shape=None, 
        gamma_cautious=(0.75, 0.75),
        lambda_cautious=(0, 0.0),
        gamma_optimistic=(0.55, 0.70),
        controller=controller,
        reset_in_safe_state=True,
        n_episodes_train=10,
        n_episodes_test=10,
        n_train_test=5,  # Number of train-test cycles (number of model checkpoints)
        # Total number of training episodes is: n_train_test * n_episodes_train
        plot_every=5  # How often figures are plotted. Low is time and memory consuming
    )
    sim.set_seed(value=seed)
    logging.info(config_msg(f'Random seed: {seed}'))
    run_t = sim.run()
    logging.info(f'Simulation duration: {run_t:.2f} s')
    sim.save_models()
