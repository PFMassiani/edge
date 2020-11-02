from pathlib import Path
import time
import logging
import numpy as np
import pandas as pd
import torch
import gc

from edge.envs.continuous_cartpole import ContinuousCartPole
from edge.simulation import ModelLearningSimulation
from edge.dataset import Dataset
from edge.utils.logging import config_msg
from edge.utils import device, cuda, timeit, average_performances,\
    append_to_episode, get_hyperparameters

# noinspection PyUnresolvedReferences
from offline_measure_agent import DLQRController, OfflineSafetyLearner


GROUP_NAME = 'Training'
OPTIM = 'Optimization'
REMAINING_T = 'Remaining time'


class OfflineMeasureSimulation(ModelLearningSimulation):
    def __init__(self, name, shape, control_frequency, perturbations,
                 max_theta_init,
                 gamma_measure,
                 n_episodes, checkpoint_dataset_every, n_optimizations,
                 render=False):
        self.env = ContinuousCartPole(
            discretization_shape=shape,  # This matters for the GP
            control_frequency=control_frequency,
            max_theta_init=max_theta_init
        )
        self.agent = DLQRController(
            env=self.env,
            perturbations=perturbations,
        )
        self.gamma_measure = gamma_measure

        output_directory = Path(__file__).parent.resolve()
        super().__init__(
            output_directory, name, {}
        )
        self.n_episodes = n_episodes
        self.checkpoint_dataset_every = checkpoint_dataset_every
        self.n_optimizations = n_optimizations
        self.render = render

        self.hyperparameters_dataset = None

    def get_last_hyperparameters(self):
        if self.hyperparameters_dataset is None:
            return None, None, None
        hp = self.hyperparameters_dataset.df
        if len(hp) == 0:
            return None, None, None
        last = hp.index[-1]
        ls = list(
            map(float, hp.loc[last, 'covar_module.base_kernel.lengthscale'])
        )
        os = float(hp.loc[last, 'covar_module.outputscale'])
        nz = float(hp.loc[last, 'likelihood.noise_covar.noise'])
        return ls, os, nz

    def create_offline_learner(self, n_optim):
        x_seed = np.array([[0, 0, 0, 0, 0.]])
        y_seed = np.array([1.])

        ls, os, nz = self.get_last_hyperparameters()
        lengthscale_means = ls if ls is not None else (
            0.1992, 1.9771, 0.2153, 2.2855, 2.1829  # From value optim
            # 0.508, 0.6, 0.12, 0.69, 0.1
        )
        lengthscale_vars = (0.1, 0.1, 0.1, 0.1, 0.1)
        lengthscale_prior = tuple(zip(lengthscale_means, lengthscale_vars))
        outputscale_prior = (os, 10.) if os is not None else (12.307, 10.)
        noise_prior = (nz, 0.1) if nz is not None else (0.716, 0.1)
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
            'dataset_params': {'append_every': 5},
            # 'dataset_type': 'neighborerasing',
            # 'dataset_params': {'radius': 0.01},
            'value_structure_discount_factor': None,
        }
        offline_learner = OfflineSafetyLearner(self.env, gp_params,
                                               self.gamma_measure)

        t = 1 if self.n_optimizations == 1 else n_optim / self.n_optimizations
        offline_learner.update_safety_params(t=t)
        return offline_learner

    def run_episode(self):
        episode = {cname: []
                   for cname in self.training_dataset.columns_wo_group}
        done = self.env.done
        while not done:
            old_state = self.agent.state
            new_state, reward, failed, done = self.agent.step()
            action = self.agent.last_action
            append_to_episode(self.training_dataset, episode, old_state, action,
                              new_state, reward, failed, done)
            if self.render:
                self.env.gym_env.render()
        return episode

    def reset_agent_state(self):
        s = self.env.s
        while self.env.done:
            s = self.agent.reset()
        return s

    def checkpoint_dataset(self, n):
        self.training_dataset.save(self.data_path / f'train_{n}.csv')

    @timeit
    def collect_dataset(self):
        for n in range(self.n_episodes):
            self.reset_agent_state()
            episode = self.run_episode()
            self.training_dataset.add_group(episode, group_number=n)
            if (n > 0) and (n % self.checkpoint_dataset_every == 0):
                self.checkpoint_dataset(n)
                self.log_performance(
                    self.training_dataset,
                    f'Checkpoint {n+1}/{self.n_episodes}',
                    header=True,
                    limit_episodes=self.checkpoint_dataset_every
                )

    def append_remaining_t(self):
        ds = self.training_dataset
        df = ds.df
        df[REMAINING_T] = df.groupby([ds.EPISODE]).cumcount(ascending=False)
        return df

    def extract_variables_from_batch(self, batch):
        episodes, rewards, states, actions, new_states, faileds, dones = batch.\
            drop(labels=[REMAINING_T], axis=1, errors='ignore').to_numpy().T
        episodes = episodes.astype(int)
        states = np.vstack(states).astype(float)
        actions = np.vstack(actions).astype(float)
        new_states = np.vstack(new_states).astype(float)
        faileds = faileds.astype(bool)
        dones = dones.astype(bool)
        return episodes, rewards, states, actions, new_states, faileds, dones

    def iteratively_set_data(self, offline_learner, df, remaining_t,
                             measures=None):
        fltr = df[REMAINING_T] == remaining_t
        batch = df.loc[fltr]
        episodes, rewards, states, actions, new_states, faileds, dones = self.\
            extract_variables_from_batch(batch)
        if measures is not None:
            measures = measures[fltr.to_numpy()]
        offline_learner.batch_update_models(states, actions, new_states,
                                            rewards, faileds, dones,
                                            measures=measures)

    def set_data(self, offline_learner, measures=None):
        if measures is None:
            df = self.append_remaining_t()
        else:
            df = self.training_dataset.df
        max_t = df[REMAINING_T].max()
        for remaining_t in range(max_t + 1):
            self.iteratively_set_data(offline_learner, df, remaining_t,
                                      measures)

    def fit_hyperparameters(self, offline_learner, n_optim):
        for lr in [1, 0.1, 0.01, 0.001]:
            offline_learner.fit_models(
                train_x=None, train_y=None,  # Fit on the GP's dataset
                epochs=20,
                lr=lr
            )
        params = get_hyperparameters(offline_learner.safety_model.gp)
        params[OPTIM] = n_optim
        if self.hyperparameters_dataset is None:
            self.hyperparameters_dataset = Dataset(
                *[cname for cname, _ in get_hyperparameters(
                    offline_learner.safety_model.gp
                ).items()],
                group_name=OPTIM,
                name='hyperparameters'
            )
        self.hyperparameters_dataset.add_entry(**params)
        self.hyperparameters_dataset.save(self.data_path)

    @timeit
    def iterate_measure(self, n_optim, previous_measures=None):
        offline_learner = self.create_offline_learner(n_optim)

        self.set_data(offline_learner, previous_measures)
        self.fit_hyperparameters(offline_learner, n_optim)

        episodes, rewards, states, actions, new_states, faileds, dones = \
            self.extract_variables_from_batch(self.training_dataset.df)
        try:
            new_measures = offline_learner.safety_model.measure(
                state=new_states
            )
        except RuntimeError as e:
            logging.error(f'Measure computation failed with error:\n{str(e)}\n'
                          f'Number of states: {len(episodes)}\nMemory status:')
            self.log_memory()
            logging.error('Re-trying with cleared cache.')
            if device == cuda:
                torch.cuda.empty_cache()
            new_measures = offline_learner.safety_model.measure(
                state=new_states)

        if previous_measures is not None:
            diff = np.linalg.norm(previous_measures - new_measures)
        else:
            diff = None

        self.save_safety_model(offline_learner, name=f'safety_model_{n_optim}')
        self.checkpoint_safety_dataset(
            n_optim, episodes, rewards, states, actions, new_states, faileds,
            dones, new_measures
        )

        return new_measures, diff

    def log_performance(self, ds, name_in_log, duration=None, header=True,
                        limit_episodes=None):
        train = ds.df
        r, f = average_performances(
            train, ds.group_name, ds.EPISODE, limit_episodes
        )
        caveat = '' if limit_episodes is None \
            else f'(last {limit_episodes} episodes) '
        header = '-------- Performance --------\n' if header else ''
        message = (f'--- {name_in_log} {caveat}\n'
                   f'Average total reward per episode: {r:.3f}\n'
                   f'Average number of failures: {f * 100:.3f} %')
        if duration is not None:
            message += f'\nComputation time: {duration:.3f} s'
        logging.info(header + message)

    def log_measures(self, measures, iterate_t, diff, n_optim, header=True):
        header = f'-------- Measure --------\n' if header else ''
        message = (f'--- Iteration {n_optim+1}/{self.n_optimizations}\n'
                   f'Average measure: {measures.mean():.4f}\n'
                   f'Nonzero measure # of points: '
                   f'{len(measures[measures>0])}/{len(measures)}\n'
                   f'Nonzero measure proportion: '
                   f'{len(measures[measures>0])/len(measures)*100:.3f} %\n')
        message += f'Measure variation: {diff:.5f}\n' if diff else ''
        message += f'Computation time: {iterate_t:.3f} s'
        logging.info(header + message)

    def log_memory(self):
        if device == cuda:
            message = ('Memory usage\n' + torch.cuda.memory_summary())
            logging.info(message)

    def log_samples(self):
        n_samples = len(self.training_dataset.df)
        logging.info(f'Training dataset size: {n_samples}')

    def checkpoint_safety_dataset(self, n_optim, *args):
        df = pd.DataFrame(
            dict(zip(
                (*Dataset.DEFAULT_COLUMNS, 'Measure'),
                map(list, args)  # pd.DataFrame requires lists, not np.ndarrays
            ))
        )
        df.to_csv(self.data_path / f'safety_{n_optim}.csv')

    def save_safety_model(self, offline_learner, name):
        savepath = self.local_models_path / 'safety_model' / name
        savepath.mkdir(exist_ok=True, parents=True)
        offline_learner.safety_model.save(savepath, save_data=True)

    def get_models_to_save(self):
        return {'safety_model': self.agent.safety_model}

    @timeit
    def run(self):
        logging.info('========= COLLECTING DATA... ========')
        try:
            col_t = self.collect_dataset()
        except RuntimeError as e:
            col_t = np.nan
            logging.critical('collect_dataset failed before completion.'
                             f'Error message:\n{e}')
        finally:
            self.log_samples()
            self.log_performance(self.training_dataset, 'All training',
                                 duration=col_t, header=True)
            logging.info('========= DATA COLLECTION COMPLETED ========')
        measures = None
        for n in range(self.n_optimizations):
            logging.info(f'========= OPTIMIZATION {n+1}/{self.n_optimizations}'
                         f' ========')
            try:
                out, iterate_t = self.iterate_measure(n, measures)
                measures, diff = out
            except RuntimeError as e:
                iterate_t = np.nan
                logging.critical(f'iterate_measure failed on iteration {n}:\n'
                                 f'{str(e)}')
                self.log_memory()
                raise e
            if device == cuda:
                torch.cuda.empty_cache()
            self.log_measures(measures, iterate_t, diff, n, header=True)
            gc.collect()
        if self.render:
            self.env.gym_env.close()


if __name__ == '__main__':
    seed = int(time.time())
    # seed = 1
    sim = OfflineMeasureSimulation(
        name=f'offline_{seed}',
        shape=(50, 50, 50, 50, 41),
        control_frequency=2,
        perturbations={'g': 1/1, 'mcart': 1, 'mpole': 1, 'l': 1},
        max_theta_init=0.5,
        gamma_measure=(0.6, 0.7),
        n_episodes=15,
        checkpoint_dataset_every=5,
        n_optimizations=10,
        render=False
    )
    sim.set_seed(value=seed)
    logging.info(config_msg(f'Random seed: {seed}'))
    run_t = sim.run()
    logging.info(f'Simulation duration: {run_t:.2f} s')
