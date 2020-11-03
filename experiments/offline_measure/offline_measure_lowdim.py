from pathlib import Path
import time
import logging
import numpy as np
import pandas as pd
import torch
import gc

from edge.simulation import ModelLearningSimulation
from edge.dataset import Dataset
from edge.utils.logging import config_msg
from edge.utils import device, cuda, timeit, average_performances,\
    append_to_episode, get_hyperparameters
from edge.agent import RandomAgent
from edge.model.safety_models import SafetyTruth
from edge.graphics.plotter import SafetyPlotter

# noinspection PyUnresolvedReferences
from offline_measure_agent import OfflineSafetyLearner
# noinspection PyUnresolvedReferences
from offline_measure_environments import LowGoalSlip, LowGoalHovership


GROUP_NAME = 'Training'
OPTIM = 'Optimization'
REMAINING_T = 'Remaining time'


class OfflineMeasureSimulation(ModelLearningSimulation):
    def __init__(self, envname, name, shape,
                 gamma_measure,
                 n_episodes, checkpoint_dataset_every, n_optimizations):
        self.envname = envname
        shapedict = {} if shape is None else {'shape': shape}
        if envname == 'hovership':
            self.env = LowGoalHovership(
                goal_state=True,
                initial_state=np.array([1.3]),
                **shapedict  # This matters for the GP
            )
        elif envname == 'slip':
            self.env = LowGoalSlip(
                initial_state=None,
                **shapedict  # This matters for the GP
            )
        else:
            raise ValueError
        self.agent = RandomAgent(
            env=self.env
        )
        self.gamma_measure = gamma_measure

        fname = 'hover_map' if envname == 'hovership' else 'slip_map'
        truth_path = Path(__file__).parent.parent.parent / 'data' / \
                     'ground_truth' / 'from_vibly' / f'{fname}.pickle'
        self.ground_truth = SafetyTruth(self.env)
        self.ground_truth.from_vibly_file(truth_path)

        output_directory = Path(__file__).parent.resolve()
        super().__init__(
            output_directory, name, {}
        )
        self.n_episodes = n_episodes
        self.checkpoint_dataset_every = checkpoint_dataset_every
        self.n_optimizations = n_optimizations

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
        t = 1 if self.n_optimizations == 1 else n_optim / self.n_optimizations

        if self.envname == 'hovership':
            x_seed = np.array([[1.3, 0.6], [0, 2]])
            y_seed = np.array([1., 0.])
            default_ls = (0.1, 0.1)
            ls_vars = (0.1, 0.1)
            default_os = 1
            default_nz = 0.1
        elif self.envname == 'slip':
            x_seed = np.array([[.45, 0.6632], [0.8, 0.4]])
            y_seed = np.array([1, 0.8])
            default_ls = (0.1, 0.1)
            ls_vars = (0.1, 0.1)
            default_os = 1
            default_nz = 0.1
        else:
            raise ValueError

        ls, os, nz = self.get_last_hyperparameters()
        lengthscale_means = ls if ls is not None else default_ls
        lengthscale_prior = tuple(zip(lengthscale_means, ls_vars))
        outputscale_prior = (os, 10.) if os is not None else (default_os, 10.)
        noise_prior = (nz, 0.1) if nz is not None else (default_nz, 0.1)
        gp_params = {
            'train_x': x_seed,
            'train_y': y_seed,
            'outputscale_prior': outputscale_prior,
            'lengthscale_prior': lengthscale_prior,
            'noise_prior': noise_prior,
            'mean_constant': 1 - t,
            # 'dataset_type': None,
            # 'dataset_params': None,
            'dataset_type': 'downsampling',
            'dataset_params': {'append_every': 2},
            # 'dataset_type': 'neighborerasing',
            # 'dataset_params': {'radius': 0.01},
            'value_structure_discount_factor': None,
        }
        offline_learner = OfflineSafetyLearner(self.env, gp_params,
                                               self.gamma_measure)

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
            if (n+1) % self.checkpoint_dataset_every == 0:
                self.checkpoint_dataset(n+1)
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
            iterable = zip(
                episodes, rewards, states, actions, new_states, faileds, dones,
                measures
            )
        else:
            iterable = zip(
                episodes, rewards, states, actions, new_states, faileds, dones
            )
        for entry in iterable:
            if measures is not None:
                e, r, s, a, s_, f, d, m = entry
            else:
                e, r, s, a, s_, f, d = entry
                m = None
            offline_learner.update_models(s, a, s_, r, f, d, m)
        # offline_learner.batch_update_models(states, actions, new_states,
        #                                     rewards, faileds, dones,
        #                                     measures=measures)

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
            new_measures = [
                offline_learner.safety_model.measure(
                    state=ns
                ) for ns in new_states.reshape(
                    -1, self.env.state_space.data_length
                )
            ]
        except RuntimeError as e:
            logging.error(f'Measure computation failed with error:\n{str(e)}\n'
                          f'Number of states: {len(episodes)}\nMemory status:')
            self.log_memory()
            logging.error('Re-trying with cleared cache.')
            if device == cuda:
                torch.cuda.empty_cache()
            new_measures = offline_learner.safety_model.measure(
                state=new_states)
        new_measures[faileds] = 0
        assert((new_measures[faileds] == 0).all())

        if previous_measures is not None:
            diff = np.linalg.norm(previous_measures - new_measures)
        else:
            diff = None

        self.save_safety_model(offline_learner, name=f'safety_model_{n_optim}')
        self.checkpoint_safety_dataset(
            n_optim, episodes, rewards, states, actions, new_states, faileds,
            dones, new_measures
        )
        self.plot_safety_measure(offline_learner, n_optim)

        return new_measures, diff

    def plot_safety_measure(self, offline_learner, n_optim):
        plotter = SafetyPlotter(offline_learner, ground_truth=self.ground_truth)
        for index, row in self.training_dataset.df.iterrows():
            _, r, s, a, s_, f, d, dt = row.to_numpy()
            plotter.on_run_iteration(s, a, s_, r, f, color=None)

        savepath = self.fig_path / f'{n_optim}_safety.pdf'
        fig = plotter.get_figure()
        fig.savefig(str(savepath), format='pdf')

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
            self.log_performance(self.training_dataset, 'All training',
                                 duration=col_t, header=True)
            self.log_samples()
            logging.info('========= DATA COLLECTION COMPLETED ========')
        measures = None
        n = 0
        diff = None
        tol = 1e-4
        while (diff is None or diff >= tol) and n < self.n_optimizations:
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
            n += 1


if __name__ == '__main__':
    # seed = int(time.time())
    seed = 1
    envname = 'hovership'
    sim = OfflineMeasureSimulation(
        envname=envname,
        name=f'offline_{envname}_{seed}_down',
        shape=None,
        gamma_measure=(0.6, 0.9),
        n_episodes=100,
        checkpoint_dataset_every=25,
        n_optimizations=15,
    )
    sim.set_seed(value=seed)
    logging.info(config_msg(f'Random seed: {seed}'))
    run_t = sim.run()
    logging.info(f'Simulation duration: {run_t:.2f} s')
