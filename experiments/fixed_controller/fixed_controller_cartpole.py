from pathlib import Path
import logging
import time
import numpy as np
import torch

from edge.envs.continuous_cartpole import ContinuousCartPole
from edge.simulation import ModelLearningSimulation
from edge.dataset import Dataset
from edge.utils.logging import config_msg
from edge.utils import device, cuda, timeit, average_performances,\
    append_to_episode as general_append

# noinspection PyUnresolvedReferences
from fixed_controller_agent import DLQRSafetyLearner


GROUP_NAME = 'Training'
SAFETY_NAME = 'Next safety'


def append_to_episode(dataset, episode, state, action, new_state, reward,
                      failed, done, safety_update):
    general_append(dataset, episode, state, action, new_state, reward,
                   failed, done)
    episode[SAFETY_NAME].append(safety_update)


def avg_reward_and_failure(df):
    r = df.groupby(Dataset.EPISODE)[Dataset.REWARD].sum().mean()
    f = df.groupby(Dataset.EPISODE)[Dataset.FAILED].any().mean()
    return r, f


class FixedControllerSimulation(ModelLearningSimulation):
    def __init__(self, name, shape, control_frequency, perturbations,
                 gamma_cautious, lambda_cautious, gamma_optimistic,
                 n_episodes_train, n_episodes_test, n_train_test,
                 load=False, load_sname=None, load_mname=None, load_dname=None,
                 render=False):
        self.env = ContinuousCartPole(
            discretization_shape=shape,  # This matters for the GP
            control_frequency=control_frequency,
        )

        if load:
            assert load_sname is not None
            assert load_mname is not None
            assert load_dname is not None
            # TODO load agent
            self.agent = self.load_agent(load_sname, load_mname)
        else:
            x_seed = np.array([[0, 0, 0, 0, 0.]])
            y_seed = np.array([1.])
            # Initialization from value GP optimization
            # TODO choose hyperparameters
            lengthscale_means = (
                # 100., 10., 10., 10., 10.
                # 0.1992, 1.9771, 0.2153, 2.2855, 2.1829  # From value optim
                0.1992, 1.9771, 0.2153, 2.2855, 2.1829  # s[0] is linear position
                # 0.508, 0.6, 0.12, 0.69, 0.1
            )
            lengthscale_vars = (0.1, 0.1, 0.1, 0.1, 0.1)
            lengthscale_prior = tuple(zip(lengthscale_means, lengthscale_vars))
            outputscale_prior = (12.30679988861084, 10.)
            noise_prior = (0.7161999940872192, 0.1)
            gp_params = {
                'train_x': x_seed,
                'train_y': y_seed,
                'outputscale_prior': outputscale_prior,
                'lengthscale_prior': lengthscale_prior,
                'noise_prior': noise_prior,
                'mean_constant': 1,
                # 'dataset_type': None,
                # 'dataset_params': None,
                'dataset_type': 'downsampling',
                'dataset_params': {'append_every': 10},
                # 'dataset_type': 'neighborerasing',
                # 'dataset_params': {'radius': 0.01},
                'value_structure_discount_factor': None,
            }
            self.agent = DLQRSafetyLearner(
                env=self.env,
                s_gp_params=gp_params,
                gamma_cautious=gamma_cautious,
                lambda_cautious=lambda_cautious,
                gamma_optimistic=gamma_optimistic,
                perturbations=perturbations,
            )

        output_directory = Path(__file__).parent.resolve()
        super(FixedControllerSimulation, self).__init__(
            output_directory, name, {}
        )
        self.n_episodes_train = n_episodes_train
        self.n_episodes_test = n_episodes_test
        self.n_train_test = n_train_test
        self.render = render

        if load:
            self.training_dataset = self.load_training_dataset(
                load_sname, load_dname, GROUP_NAME
            )
        else:
            self.training_dataset = Dataset(
                *Dataset.DEFAULT_COLUMNS, SAFETY_NAME,
                group_name=GROUP_NAME, name='train'
            )
        self.testing_dataset = Dataset(
                *Dataset.DEFAULT_COLUMNS, SAFETY_NAME,
                group_name=GROUP_NAME, name=f'test'
        )

    def run_episode(self, n_episode):
        episode = {cname: []
                   for cname in self.training_dataset.columns_wo_group}
        done = self.env.done
        while not done:
            old_state = self.agent.state
            new_state, reward, failed, done = self.agent.step()
            action = self.agent.last_action
            safety_update = self.agent.safety_update if self.agent.do_safety_update else None
            append_to_episode(self.training_dataset, episode, old_state, action,
                              new_state, reward, failed, done, safety_update)
            if self.render:
                self.env.gym_env.render()
        len_episode = len(episode[self.training_dataset.REWARD])
        episode[self.training_dataset.EPISODE] = [n_episode] * len_episode
        return episode

    def reset_agent_state(self):
        s = self.env.s
        while self.env.done:
            s = self.agent.reset()
        return s

    @timeit
    def train_agent(self, n_train):
        self.agent.training_mode = True
        for n in range(self.n_episodes_train):
            self.reset_agent_state()
            episode = self.run_episode(n)
            self.training_dataset.add_group(episode, group_number=None)

    @timeit
    def test_agent(self, n_test):
        self.agent.training_mode = False
        for n in range(self.n_episodes_test):
            self.reset_agent_state()
            episode = self.run_episode(n)
            self.testing_dataset.add_group(episode, group_number=n_test)

    @timeit
    def log_performance(self, n_train, ds, name_in_log, duration, header=True,
                        limit_episodes=None):
        df = ds.df
        train = df.loc[df[ds.group_name] == n_train, :]
        r, f = average_performances(
            train, ds.group_name, ds.EPISODE, limit_episodes
        )
        caveat = '' if limit_episodes is None \
            else f'(last {limit_episodes} episodes) '
        header = '-------- Performance --------\n' if header else ''
        message = (f'--- {name_in_log} {caveat}\n'
                   f'Average total reward per episode: {r:.3f}\n'
                   f'Average number of failures: {f * 100:.3f} %\n'
                   f'Computation time: {duration:.3f} s')
        logging.info(header + message)

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
            t = 1 if self.n_train_test == 1 else n / (self.n_train_test - 1)
            self.agent.update_safety_params(t=t)
            try:
                train_t = self.train_agent(n)
            except RuntimeError as e:
                train_t = np.nan
                logging.critical(f'train_agent({n}) failed:\n{str(e)}')
                self.log_memory()
                torch.cuda.empty_cache()
            finally:
                self.log_performance(n, self.training_dataset, 'Training',
                                     train_t, header=True, limit_episodes=1)
            self.log_samples()
            try:
                test_t = self.test_agent(n)
            except RuntimeError as e:
                test_t = np.nan
                logging.critical(f'test_agent({n}) failed:\n{str(e)}')
                torch.cuda.empty_cache()
            finally:
                self.log_performance(n, self.testing_dataset, 'Testing',
                                     test_t, header=False, limit_episodes=None)
            chkpt_t = self.checkpoint(n)
            logging.info(f'Checkpointing time: {chkpt_t:.3f} s')
        if self.render:
            self.env.gym_env.close()


if __name__ == '__main__':
    # seed = int(time.time())
    seed = 0
    sim = FixedControllerSimulation(
        name=f'cartpole_{seed}',
        shape=(10, 10, 10, 10, 41),
        control_frequency=2,
        perturbations={'g': 1, 'mcart': 1, 'mpole': 1, 'l': 1/5},
        gamma_cautious=(0.6, 0.7),
        lambda_cautious=(0, 0.05),
        gamma_optimistic=(0.55, 0.7),
        n_episodes_train=1,
        n_episodes_test=0,
        n_train_test=5,
        load=False,
        load_sname=None,
        load_mname=None,
        load_dname=None,
        render=True
    )
    sim.set_seed(value=seed)
    logging.info(config_msg(f'Random seed: {seed}'))
    run_t = sim.run()
    logging.info(f'Simulation duration: {run_t:.2f} s')
    sim.save_models()
