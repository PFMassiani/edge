from pathlib import Path
import logging
import time
import numpy as np
import torch

from edge.simulation import ModelLearningSimulation
from edge.graphics.plotter import SafetyPlotter
from edge.dataset import Dataset
from edge.utils.logging import config_msg
from edge.utils import device, cuda, timeit, append_to_episode as general_append
from edge.model.safety_models import SafetyTruth

# noinspection PyUnresolvedReferences
from fixed_controller_agent import RandomSafetyLearner, FreeRandomSafetyLearner
# noinspection PyUnresolvedReferences
from fixed_controller_environments import LowGoalHovership, LowGoalSlip


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


class FixedControllerLowdim(ModelLearningSimulation):
    def __init__(self, envname, name, shape,
                 gamma_cautious, lambda_cautious, gamma_optimistic,
                 model_type, discount,
                 n_episodes_train, n_episodes_test, n_train_test,
                 load=False, load_sname=None, load_mname=None, load_dname=None,
                 plot_every=1):
        shapedict = {} if shape is None else {'shape': shape}
        if envname == 'hovership':
            self.env = LowGoalHovership(
                goal_state=True,
                initial_state=np.array([1.3]),
                **shapedict  # This matters for the GP
            )
        elif envname == 'slip':
            self.env = LowGoalSlip(
                initial_state=np.array([.45]),
                **shapedict  # This matters for the GP
            )
        else:
            raise ValueError

        if load:
            assert load_sname is not None
            assert load_mname is not None
            assert load_dname is not None
            # TODO load agent
            self.agent = self.load_agent(load_sname, load_mname)
        elif envname == 'hovership':
            x_seed = np.array([[1.3, 0.6], [0, 2]])
            y_seed = np.array([1., 0.])
            lengthscale_means = (0.1, 0.1)
            lengthscale_vars = (0.1, 0.1)
            lengthscale_prior = tuple(zip(lengthscale_means, lengthscale_vars))
            outputscale_prior = (1., 10.)
            noise_prior = (0.1, 0.1)
            discount = discount if model_type == 'td' else None
        elif envname == 'slip':
            x_seed = np.array([[.45, 0.6632], [0.8, 0.4]])
            y_seed = np.array([1, 0.8])
            lengthscale_means = (0.1, 0.1)
            lengthscale_vars = (0.1, 0.1)
            lengthscale_prior = tuple(zip(lengthscale_means, lengthscale_vars))
            outputscale_prior = (1., 10.)
            noise_prior = (0.1, 0.1)
            discount = discount if model_type == 'td' else None
        else:
            raise ValueError

        if not load:
            gp_params = {
                'train_x': x_seed,
                'train_y': y_seed,
                'outputscale_prior': outputscale_prior,
                'lengthscale_prior': lengthscale_prior,
                'noise_prior': noise_prior,
                # 'dataset_type': None,
                # 'dataset_params': None,
                # 'dataset_type': 'downsampling',
                # 'dataset_params': {'append_every': 10},
                'dataset_type': 'neighborerasing',
                'dataset_params': {'radius': 0.01},
                'value_structure_discount_factor': discount,
            }
            self.agent = RandomSafetyLearner(
                env=self.env,
                s_gp_params=gp_params,
                gamma_cautious=gamma_cautious,
                lambda_cautious=lambda_cautious,
                gamma_optimistic=gamma_optimistic,
                model_type=model_type,
            )

        fname = 'hover_map' if envname == 'hovership' else 'slip_map'
        truth_path = Path(__file__).parent.parent.parent / 'data' / \
                     'ground_truth' / 'from_vibly' / f'{fname}.pickle'
        ground_truth = SafetyTruth(self.env)
        ground_truth.from_vibly_file(truth_path)
        plotters = {
            'safety': SafetyPlotter(self.agent, ground_truth=ground_truth)
        }

        output_directory = Path(__file__).parent.resolve()
        super().__init__(output_directory, name, plotters)

        self.n_episodes_train = n_episodes_train
        self.n_episodes_test = n_episodes_test
        self.n_train_test = n_train_test
        self.plot_every = plot_every

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
            safety_update = self.agent.safety_update
            append_to_episode(self.training_dataset, episode, old_state, action,
                              new_state, reward, failed, done, safety_update)
            if self.agent.training_mode:
                color = [1., 0., 0.] if self.agent.do_safety_update \
                                     else [0., 1., 0.]
                super().on_run_iteration(state=old_state, action=action,
                                         new_state=new_state, reward=reward,
                                         failed=failed, color=color)
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
            if n % self.plot_every == 0:
                self.save_figs(prefix=f'{n_train}ep{n}')

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
        # TODO define useful performances
        df = ds.df
        train = df.loc[df[ds.group_name] == n_train, :]
        n_episodes = train.loc[:, ds.EPISODE].max()
        if limit_episodes is None:
            limit_episodes = n_episodes
        r, f = avg_reward_and_failure(
            train.loc[train[ds.EPISODE] >= n_episodes - limit_episodes])

        header = '-------- Performance --------\n' if header else ''
        message = (f'--- {name_in_log}\n'
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
            t = 1 if self.n_train_test == 1 else n/(self.n_train_test - 1)
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
                                     train_t, header=True, limit_episodes=10)
            self.log_samples()
            # try:
            #     test_t = self.test_agent(n)
            # except RuntimeError as e:
            #     test_t = np.nan
            #     logging.critical(f'test_agent({n}) failed:\n{str(e)}')
            #     torch.cuda.empty_cache()
            # finally:
            #     self.log_performance(n, self.testing_dataset, 'Testing',
            #                          test_t, header=False, limit_episodes=None)
            chkpt_t = self.checkpoint(n)
            logging.info(f'Checkpointing time: {chkpt_t:.3f} s')


if __name__ == '__main__':
    # seed = int(time.time())
    seed = 2
    envname = 'hovership'
    model_type = 'default'
    sim = FixedControllerLowdim(
        envname=envname,
        name=f'{envname}_{model_type}_{seed}',
        shape=None,  # (201, 161),
        gamma_cautious=(0.6, 0.6),
        lambda_cautious=(0, 0.0),
        gamma_optimistic=(0.55, 0.55),
        model_type=model_type,
        discount=None,
        n_episodes_train=100,
        n_episodes_test=0,
        n_train_test=1,
        load=False,
        load_sname=None,
        load_mname=None,
        load_dname=None,
        plot_every=5
    )
    sim.set_seed(value=seed)
    logging.info(config_msg(f'Random seed: {seed}'))
    run_t = sim.run()
    logging.info(f'Simulation duration: {run_t:.2f} s')
    sim.save_models()
