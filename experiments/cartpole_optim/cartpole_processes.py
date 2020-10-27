# train an agent with a set of hyperparameters
# evaluate the learned policy (total reward)
# optimize the hyperparameters on that set
# update the agent's hyperparameters with the result of the optimization
# reset the agent's dataset
# re-do the previous steps until satisfactory performance is reached
# plots: reward = f(hyperparameters): MC evaluation of the policy at the
#                   end of the training and before optimization
#             hyperparameters = f(n of trainings)
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

N = 5
GAMMA = 0.99
XI = 0.01
TRAINING = 'Training number'
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


class CartPoleProcess(ModelLearningSimulation):
    def __init__(self, name, penalty, control_frequency,
                 lengthscales_means, outputscale_mean, noise_mean,
                 n_episodes_train, n_episodes_test, n_training):
        self.env = ContinuousCartPole(
            discretization_shape=(10, 10, 10, 10, 10),  # This does not matter
            control_frequency=control_frequency,
        )
        # STATE SPACE:
        # X, Y, V_X, V_Y, THETA, dTHETA/dt, (CTCT_L, CTCT_R)
        # Note: the last two dimensions are removed compare to standard Gym
        self.x_seed = np.array([
            [   0,   0,     0,    0, 0],
            [-0.1, -0.1, 0.05, 0.01, 0]
        ])
        self.y_seed = np.array([1, 1])
        # Initialization from previous optimization
        self.lengthscale_means = lengthscales_means
        self.outputscale_mean = outputscale_mean
        self.noise_mean = noise_mean
        self.agent = None
        self.create_new_agent()

        output_directory = Path(__file__).parent.resolve()
        super(CartPoleProcess, self).__init__(
            output_directory, name, {}
        )

        self.n_episodes_train = n_episodes_train
        self.n_episodes_test = n_episodes_test
        self.n_training = n_training

        self.lr = 0.1

        self.training_dataset = Dataset(group_name=TRAINING, name=f'train_{n_training}')
        self.testing_dataset = Dataset(group_name=TRAINING, name=f'test_{n_training}')
        self.hyperparameters_dataset = Dataset(
            *[cname for cname, _ in get_hyperparameters(self.agent.Q_model.gp
                                                        ).items()],
            group_name=TRAINING,
            name=f'hyperparameters_{n_training}'
        )

        lengthscale_vars = (0.01, 0.01, 0.01, 0.01, 0.01)
        outputscale_var = 10.
        noise_var = 0.1
        outputscale_prior = (self.outputscale_mean, outputscale_var)
        lengthscale_prior = tuple(zip(self.lengthscale_means, lengthscale_vars))
        noise_prior = (self.noise_mean, noise_var)
        gp_params = {
            'train_x': self.x_seed,
            'train_y': self.y_seed,
            'outputscale_prior': outputscale_prior,
            'lengthscale_prior': lengthscale_prior,
            'noise_prior': noise_prior,
            'dataset_type': None,
            'dataset_params': None,
            'value_structure_discount_factor': GAMMA,
        }
 
        self.agent = CartpoleSARSALearner(
            env=self.env,
            xi=XI,
            keep_seed_in_data=True,
            q_gp_params=gp_params,
        )

    def create_new_agent(self):
        outputscale_var = 10.
        lengthscale_vars = (0.1, 0.1, 0.1, 0.1, 0.1)
        noise_var = 0.1
        outputscale_prior = (self.outputscale_mean, outputscale_var)
        lengthscale_prior = tuple(zip(self.lengthscale_means, lengthscale_vars))
        noise_prior = (self.noise_mean, noise_var)
        gp_params = {
            'train_x': self.x_seed,
            'train_y': self.y_seed,
            'outputscale_prior': outputscale_prior,
            'lengthscale_prior': lengthscale_prior,
            'noise_prior': noise_prior,
            'dataset_type': None,
            'dataset_params': None,
            'value_structure_discount_factor': GAMMA,
        }
        self.log_memory() 
        self.agent = CartpoleSARSALearner(
             env=self.env,
             xi=XI,
             keep_seed_in_data=True,
             q_gp_params=gp_params,
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
    def fit_hyperparameters(self, n_train):
        self.agent.fit_models(
            train_x=None, train_y=None,  # Fit on the GP's dataset
            epochs=200,
            lr=self.lr
        )
        self.lr /= 5
        params = get_hyperparameters(self.agent.Q_model.gp)
        params[TRAINING] = n_train
        self.hyperparameters_dataset.add_entry(**params)

    def log_hyperparameters(self, fit_t):
        message = '-------- Hyperparameters --------'
        tab = ''
        headers = ['    Parameter', '    Value (transformed)', '    Constraint']
        params = [[pname, str(pval[0]), str(pval[1])]
                  for pname, pval in get_hyperparameters(self.agent.Q_model.gp,
                                                         constraints=True
                                                         ).items()]
        rows = [headers] + params
        cols = list(zip(*rows))  # List transposition
        cols_lens = [len(str(max(col, key=len))) for col in cols]
        # This creates a table with consistent column width
        for row in rows:
            tab += '\n'
            tab += '|'.join(
                '{0:{width}}'.format(item, width=clen)
                for item, clen in zip(row, cols_lens)
            )
        message += tab
        message += f'\nHyperparameters fitting computation time: {fit_t:.3f} s'
        logging.info(message)

    @timeit
    def log_performance(self, n_train, ds, name_in_log, duration, header=True):
        df = ds.df
        train = df.loc[df[ds.group_name] == n_train, :]
        r, f = avg_reward_and_failure(train)

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
        self.hyperparameters_dataset.save(self.data_path)
        self.save_q_model(f'Q_model_{n}')

    @timeit
    def run(self):
        logging.info(f'======== TRAINING {self.n_training}/{N} ========')
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
                                 train_t, True)
        try:
            test_t = self.test_agent(n)
        except RuntimeError as e:
            test_t = np.nan
            logging.critical(f'test_agent({n}) failed:\n{str(e)}')
            torch.cuda.empty_cache()
        finally:
            self.log_performance(n, self.testing_dataset, 'Testing',
                                 test_t, False)
        try:
            fit_t = self.fit_hyperparameters(n)
        except RuntimeError as e:
            logging.critical(f'fit_hyperparameters({n}) failed:\n{str(e)}')
            fit_t = np.nan
        self.log_hyperparameters(fit_t)
        chkpt_t = self.checkpoint(n)
        logging.info(f'Checkpointing time: {chkpt_t:.3f} s')

    def load_models(self, skip_local=False):
        if not skip_local:
            load_path = self.local_models_path / 'Q_model'
        else:
            load_path = self.models_path / 'Q_model'
        self.agent.Q_model = MaternGPSARSA.load(load_path, self.env,
                                                self.x_seed, self.y_seed)

    def save_q_model(self, name):
        savepath = self.local_models_path / 'Q_model' / name
        savepath.mkdir(exist_ok=True, parents=True)
        self.agent.Q_model.save(savepath, save_data=True)

    def get_models_to_save(self):
        return {'Q_model': self.agent.Q_model}


if __name__ == '__main__':
    name_seed = int(time.time())
    lengthscales = (0.508, 0.6, 0.12, 0.69, 0.1)
    outputscale = 34.
    noise = 0.139 
    print(config_msg(f'Name seed: {name_seed}'))
    for n in range(N):
        seed = int(time.time())
        sim = CartPoleProcess(
            name=f'process_{name_seed}',
            penalty=None,
            control_frequency=2,  # Higher increases the number of possible episodes
            lengthscales_means=lengthscales,
            outputscale_mean=outputscale,
            noise_mean=noise,
            n_episodes_train=70,
            n_episodes_test=30,
            n_training=n
        )
        logging.info(config_msg(f'Random seed: {seed}'))
        sim.set_seed(value=seed)
        run_t = sim.run()
        logging.info(f'Simulation duration: {run_t:.2f} s')
        hyper = get_hyperparameters(sim.agent.Q_model.gp, constraints=False)
        ls_key = [k for k in hyper.keys() if 'length' in k][0]
        os_key = [k for k in hyper.keys() if 'output' in k][0]
        nz_key = [k for k in hyper.keys() if 'noise' in k][0]
        lengthscales = hyper[ls_key]
        ouputscale = hyper[os_key]
        noise = hyper[nz_key]
        del sim
        gc.collect()
