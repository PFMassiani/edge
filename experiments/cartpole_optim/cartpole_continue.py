from pathlib import Path
import numpy as np
import logging
import time
import functools
from gpytorch import settings
import torch

from edge.simulation import ModelLearningSimulation
from edge.dataset import Dataset
from edge.model.value_models import MaternGPSARSA
from edge.envs.continuous_cartpole import ContinuousCartPole
from edge.utils import cuda, device
from edge.utils.logging import config_msg

# noinspection PyUnresolvedReferences
from cartpole_agent import CartpoleSARSALearner


CYCLE = 'Cycle'
TRAINING = 'Training number'
DEBUG = False
FAST_COMPS_SOLVES = True


class MaternGPSARSALoadingSimulation(ModelLearningSimulation):
    def load_agent(self, env, xi, sname, mname):
        # TODO add case where the Agent has a safety model
        self.agent = CartpoleSARSALearner(
            env=env,
            xi=xi,
            keep_seed_in_data=True,
            q_gp_params={
                'train_x': np.array([[0, 0, 0, 0, 0.] * 2]),
                'train_y': np.array([0, 0.]),
                'value_structure_discount_factor': 0.9,
            },
            s_gp_params=None,
            gamma_cautious=None,
            lambda_cautious=None,
            gamma_optimistic=None
        )

        load_folder = Path(__file__).parent.resolve() / sname / 'models' / \
                      'Q_model' / mname
        self.agent.Q_model = MaternGPSARSA.load(
            env=env,
            load_folder=load_folder,
            x_seed=np.array([[0, 0, 0, 0, 0.] * 2]),
            y_seed=np.array([0, 0.]),
            load_data=True
        )

    def load_training_dataset(self, sname, dname, group=None):
        dpath = Path(__file__).parent.resolve() / sname / 'data' / dname
        ds = Dataset.load(dpath, group_name=GROUP_NAME)
        if group is not None:
            ds.df = ds.df.loc[ds.df.loc[:, ds.group_name] == group]
        self.training_dataset = ds

    def save_q_model(self, name):
        savepath = self.local_models_path / 'Q_model' / name
        savepath.mkdir(exist_ok=True, parents=True)
        self.agent.Q_model.save(savepath, save_data=True)


def timeit(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        try:
            out = f(*args, **kwargs)
            t1 = time.time()
        except Exception as e:
            t1 = time.time()
            print(f'Function {f.__name__} failed after {t1 - t0:.3f} s')
            if DEBUG:
                print('Entering debugging mode')
                import pudb
                pudb.post_mortem()
            raise e
        if out is None:
            return t1 - t0
        else:
            return out, t1 - t0
    return wrapper

def append_to_episode(dataset, episode, state, action, new_state, reward,
                      failed, done):
    episode[dataset.STATE].append(state)
    episode[dataset.ACTION].append(action)
    episode[dataset.NEW].append(new_state)
    episode[dataset.REWARD].append(reward)
    episode[dataset.FAILED].append(failed)
    episode[dataset.DONE].append(done)


def get_hyperparameters(gp, constraints=False):
    params = {}
    for name, param, constraint \
            in gp.named_parameters_and_constraints():
        transformed = np.around(
            constraint.transform(param).cpu().detach().numpy().squeeze(),
            decimals=4
        )
        entry = transformed if not constraints else (transformed, constraint)
        key = ''.join(name.split('raw_'))
        params[key] = entry
    return params


def avg_reward_and_failure(df):
    r = df.groupby(Dataset.EPISODE)[Dataset.REWARD].sum().mean()
    f = df.groupby(Dataset.EPISODE)[Dataset.FAILED].any().mean()
    return r, f


class CartPoleLoading(MaternGPSARSALoadingSimulation):
    def __init__(self, name, shape, penalty, xi, control_frequency,
                 load_sname, load_mname, load_dataset_name, load_group,
                 n_episodes_train, n_episodes_test, n_train_test):
        self.env = ContinuousCartPole(
            discretization_shape=shape,  # This does not matter
            control_frequency=control_frequency,
        )
        self.xi = xi

        self.agent = None
        self.load_agent(self.env, xi, load_sname, load_mname)

        output_directory = Path(__file__).parent.resolve()
        super(CartPoleLoading, self).__init__(
            output_directory, name, {}
        )

        self.n_episodes_train = n_episodes_train
        self.n_episodes_test = n_episodes_test
        self.n_train_test = n_train_test

        self.load_training_dataset(load_sname, load_dataset_name, load_group)
        self.testing_dataset = Dataset(group_name=GROUP_NAME, name=f'test')

    def adapt_training_dataset(self):
        # Reset episodes
        ds = self.training_dataset
        df = self.training_dataset.df
        df[ds.EPISODE] = df[ds.EPISODE].diff().fillna(value=0).\
            astype(bool).astype(int).cumsum()
        # Remove group_name if not episode
        if ds.group_name != ds.EPISODE:
            ds.df = df.drop(columns=[ds.group_name])
            ds.columns = ds.columns_wo_group
            ds.columns_wo_group = [cname for cname in ds.columns
                                   if cname != ds.EPISODE]

    def run_episode(self, n_episode):
        with settings.fast_computations(solves=FAST_COMPS_SOLVES):
            episode = {cname: []
                       for cname in self.training_dataset.columns_wo_group}
            done = self.env.done
            while not done:
                old_state = self.agent.state
                new_state, reward, failed, done = self.agent.step()
                action = self.agent.last_action
                append_to_episode(self.training_dataset, episode, old_state, action,
                                  new_state, reward, failed, done)
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
    def log_performance(self, n_train, ds, name_in_log, duration, header=True, limit_episodes=None):
        # TODO define useful performances
        df = ds.df
        train = df.loc[df[ds.group_name] == n_train, :]
        if limit_episodes is None:
            limit_episodes = len(train)
        r, f = avg_reward_and_failure(train.loc[train[ds.EPISODE] >= n_train - limit_episodes])

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
        n_samples = self.agent.Q_model.gp.train_x.shape[0]
        logging.info(f'Training dataset size: {n_samples}')

    @timeit
    def checkpoint(self, n):
        self.training_dataset.save(self.data_path)
        self.testing_dataset.save(self.data_path)
        self.save_q_model(f'Q_model_{n}')

    @timeit
    def run(self):
        for n in range(self.n_train_test):
            logging.info(f'========= CYCLE {n+1}/{self.n_train_test} ========')
            self.log_samples()
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
        self.env.gym_env.close()

    def load_models(self, skip_local=False):
        pass

    def get_models_to_save(self):
        return {'Q_model': self.agent.Q_model}


if __name__ == '__main__':
    GROUP_NAME = TRAINING
    seed = int(time.time())
    load_seed = 1603450444
    group_num = 4
    sim = CartPoleLoading(
        name=f'render',#_{seed}',
        shape=(10, 10, 10, 10, 10),
        penalty=None,
        xi=0.01,
        control_frequency=2,
        load_sname=f'process_saved_{load_seed}',
        load_mname=f'Q_model_{group_num}',
        load_dataset_name=f'train_{group_num}.csv',
        load_group=group_num,
        n_episodes_train=0,
        n_episodes_test=30,
        n_train_test=1
    )
    sim.set_seed(value=seed)
    logging.info(config_msg(f'Random seed: {seed}'))
    run_t = sim.run()
    logging.info(f'Simulation duration: {run_t:.2f} s')
    sim.save_models()
