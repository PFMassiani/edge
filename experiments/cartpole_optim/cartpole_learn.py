import time
import functools
import gc
import numpy as np
from pathlib import Path
import logging
from gpytorch import settings
import torch

from edge.simulation import ModelLearningSimulation
from edge.dataset import Dataset
from edge.model.value_models import MaternGPSARSA
from edge.utils.logging import config_msg
from edge.utils import device, cuda
from edge.envs.continuous_cartpole import ContinuousCartPole

# noinspection PyUnresolvedReferences
from cartpole_agent import CartpoleSARSALearner

CYCLE = 'Cycle'
DEBUG = False 
FAST_COMPS_SOLVES = True


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


class CartPoleLearning(ModelLearningSimulation):
    def __init__(self, name, shape, penalty, xi, control_frequency,
                 lengthscales_means, outputscale_mean, noise_mean,
                 n_episodes_train, n_episodes_test, n_train_test,
                 discount_rate, q_x_seed, q_y_seed,
                 use_safety_model, s_x_seed, s_y_seed,
                 gamma_cautious, lambda_cautious, gamma_optimistic,
                 ):
        self.env = ContinuousCartPole(
            discretization_shape=(10, 10, 10, 10, 10),  # This does not matter
            control_frequency=control_frequency,
        )
        self.xi = xi
        self.q_x_seed = q_x_seed
        self.q_y_seed = q_y_seed
        self.discount_rate = discount_rate
        self.s_x_seed = s_x_seed
        self.s_y_seed = s_y_seed
        self.use_safety_model = use_safety_model
        self.gamma_cautious = gamma_cautious
        self.lambda_cautious = lambda_cautious
        self.gamma_optimistic = gamma_optimistic

        self.lengthscale_means = lengthscales_means
        self.outputscale_mean = outputscale_mean
        self.noise_mean = noise_mean

        self.agent = None
        self.create_new_agent()

        output_directory = Path(__file__).parent.resolve()
        super(CartPoleLearning, self).__init__(
            output_directory, name, {}
        )

        self.n_episodes_train = n_episodes_train
        self.n_episodes_test = n_episodes_test
        self.n_train_test = n_train_test

        self.training_dataset = Dataset(group_name=CYCLE, name=f'train')
        self.testing_dataset = Dataset(group_name=CYCLE, name=f'test')

    def create_new_agent(self):
        outputscale_var = 10.
        lengthscale_vars = (0.1, 0.1, 0.1, 0.1, 0.1)
        noise_var = 0.1
        outputscale_prior = (self.outputscale_mean, outputscale_var)
        lengthscale_prior = tuple(zip(self.lengthscale_means, lengthscale_vars))
        noise_prior = (self.noise_mean, noise_var)
        q_gp_params = {
            'train_x': self.q_x_seed,
            'train_y': self.q_y_seed,
            'outputscale_prior': outputscale_prior,
            'lengthscale_prior': lengthscale_prior,
            'noise_prior': noise_prior,
            'dataset_type': None,
            'dataset_params': None,
            'value_structure_discount_factor': self.discount_rate,
        }
        s_gp_params = {
            'train_x': self.s_x_seed,
            'train_y': self.s_y_seed,
            'outputscale_prior': outputscale_prior,
            'lengthscale_prior': lengthscale_prior,
            'noise_prior': noise_prior,
            'dataset_type': None,
            'dataset_params': None,
            'value_structure_discount_factor': None,
        }
        self.log_memory() 
        self.agent = CartpoleSARSALearner(
            env=self.env,
            xi=self.xi,
            keep_seed_in_data=True,
            q_gp_params=q_gp_params,
            s_gp_params=s_gp_params if self.use_safety_model else None,
            gamma_cautious=self.gamma_cautious if self.use_safety_model else None,
            lambda_cautious=self.lambda_cautious if self.use_safety_model else None,
            gamma_optimistic=self.gamma_optimistic if self.use_safety_model else None
        )

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
            self.training_dataset.add_group(episode, group_number=n_train)

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

    @timeit
    def checkpoint(self, n):
        self.training_dataset.save(self.data_path)
        self.testing_dataset.save(self.data_path)
        self.save_q_model(f'Q_model_{n}')

    @timeit
    def run(self):
        for n in range(self.n_train_test):
            logging.info(f'========= CYCLE {n+1}/{self.n_train_test} ========')
            try:
                train_t = self.train_agent(n)
            except RuntimeError as e:
                train_t = np.nan
                logging.critical(f'train_agent({n}) failed:\n{str(e)}')
                self.log_memory()
                if DEBUG:
                    import pudb
                    pudb.post_mortem()
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

    def load_models(self, skip_local=False):
        if not skip_local:
            load_path = self.local_models_path / 'Q_model'
        else:
            load_path = self.models_path / 'Q_model'
        self.agent.Q_model = MaternGPSARSA.load(load_path, self.env,
                                                self.q_x_seed, self.q_y_seed)

    def save_q_model(self, name):
        savepath = self.local_models_path / 'Q_model' / name
        savepath.mkdir(exist_ok=True, parents=True)
        self.agent.Q_model.save(savepath, save_data=True)


if __name__ == '__main__':
    q_x_seed = np.array([
        [   0,   0,     0,    0, 0],
        [-0.1, -0.1, 0.05, 0.01, 0]
    ])
    q_y_seed = np.array([1, 1])
    s_x_seed = np.array([
        [0, 0,    0, 0, 0],
        [0, 0, -0.4, 0, 0],
        [0, 0,  0.4, 0, 0]
    ])
    s_y_seed = np.array([10, 0.1, 0.1])

    lengthscales = (0.508, 0.6, 0.12, 0.69, 0.1)
    outputscale = 34.
    noise = 0.139 

    seed = int(time.time())
    sim = CartPoleLearning(
        name=f'learning_{seed}',
        shape=(10, 10, 10, 10, 10),
        penalty=None,
        xi=0.01,
        control_frequency=2,  # Higher increases the number of possible episodes
        lengthscales_means=lengthscales,
        outputscale_mean=outputscale,
        noise_mean=noise,
        n_episodes_train=4,
        n_episodes_test=1,
        n_train_test=1,
        discount_rate=0.9,
        q_x_seed=q_x_seed,
        q_y_seed=q_y_seed,
        use_safety_model=False,
        s_x_seed=s_x_seed,
        s_y_seed=s_y_seed,
        gamma_cautious=0.75,  # TODO vary during simulation
        lambda_cautious=0.05,  # TODO vary during simulation
        gamma_optimistic=0.6  # TODO vary during simulation
    )

    logging.info(config_msg(f'Random seed: {seed}'))
    sim.set_seed(value=seed)
    run_t = sim.run()
    logging.info(f'Done.\nSimulation duration: {run_t:.2f} s')

