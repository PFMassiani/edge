import argparse
from pathlib import Path
import numpy as np
import logging

from edge.envs.continuous_cartpole import ContinuousCartPole
from edge.dataset import Dataset
from edge.utils import append_to_episode as general_append, \
    average_performances as general_performances
from edge.utils.logging import setup_default_logging_configuration, \
    reset_default_logging_configuration

from learned_mean_agent import DLQRSafetyLearner
from offline_measure_agent import DLQRController


SAFETY_NAME = 'Next safety'
MODELNUM_NAME = 'Model number'


def average_performances(ds):
    df = ds.df
    r, f = general_performances(
        df, ds.group_name, ds.EPISODE, None
    )
    # s = df.groupby(ds.group_name)[SAFETY_NAME].mean().mean()
    return r, f


def append_to_episode(dataset, episode, state, action, new_state, reward,
                      failed, done, safety_update):
    general_append(dataset, episode, state, action, new_state, reward,
                   failed, done)
    # episode[SAFETY_NAME].append(safety_update)


def run_episode(agent, ds, render):
    episode = {cname: []
               for cname in ds.columns_wo_group}
    while agent.env.done:
        agent.reset()
    done = agent.env.done
    while not done:
        old_state = agent.state
        new_state, reward, failed, done = agent.step()
        action = agent.last_action
        try:
            safety_update = agent.safety_update if agent.do_safety_update \
                else None
        except AttributeError:
            safety_update = None
        append_to_episode(ds, episode, old_state, action,
                          new_state, reward, failed, done, safety_update)
        if render:
            agent.env.gym_env.render()
    return episode


def run(agent, max_episodes, render, log_every, performances, modelnum):
    # ds = Dataset(*Dataset.DEFAULT_COLUMNS, SAFETY_NAME)
    # performances = Dataset(Dataset.REWARD, Dataset.FAILED, name=name)
    ds = Dataset()
    for n_episode in range(max_episodes):
        episode = run_episode(agent, ds, render)
        ds.add_group(episode, group_number=n_episode)
        if (n_episode + 1) % log_every == 0:
            r, f = log_performance(ds, n_episode+1, max_episodes)
            performances.add_entry(modelnum, n_episode, r, f)
    if render:
        agent.env.gym_env.close()


def log_performance(ds, n_episode, max_episodes):
    r, f = average_performances(ds)
    header = f'------ Checkpoint {n_episode}/{max_episodes} ------\n'
    message = (f'Average total reward per episode: {r:.3f}\n'
               f'Average number of failures: {f * 100:.3f} %\n')
               # f'Average next state safety: {s:.3f} %\n')
    logging.info(header + message)
    return r, f


def get_t_from_modelnum(modelnum, max_modelnum):
    return 1 if max_modelnum == 0 else modelnum / max_modelnum


def modelnum_iter(modelspath):
    for mpath in modelspath.iterdir():
        modelname = mpath.stem
        modelnum = int(modelname.split('_')[-1])
        yield modelnum


def offline_path(offline_seed):
    return Path(__file__).parent.resolve() / f'offline_{offline_seed}'


def learned_path(offline_seed):
    return offline_path(offline_seed) / 'models' / 'learned_mean'


def learned_load_path(offline_seed, modelnum):
    learned_safety_path = (learned_path(offline_seed) /
                           f'learned_mean_{offline_seed}_{modelnum}' /
                           'models' / 'safety_model')
    max_modelnum = max(modelnum_iter(learned_safety_path))
    return learned_safety_path / f'safety_model_{max_modelnum}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('offline_seed')
    parser.add_argument('--episodes', default=6)
    parser.add_argument('--render', default=False)
    parser.add_argument('--log', default=2)

    args = parser.parse_args()

    setup_default_logging_configuration(
        log_path=offline_path(args.offline_seed) / 'logs' / 'evaluation.log'
    )

    n_safety_params_updates = 100
    gamma_cautious = (0.7, 0.95)
    lambda_cautious = (0., 0.05)
    gamma_optimistic = (0.6, 0.9)
    max_theta_init = 0.4
    shape = (50, 50, 50, 50, 41)
    perturbations = {'g': 1/1, 'mcart': 1, 'mpole': 1, 'l': 1}
    control_frequency = 2
    x_seed = np.array([[0, 0, 0, 0, 0.]])
    y_seed = np.array([1.])

    performances = Dataset(Dataset.EPISODE, Dataset.REWARD, Dataset.FAILED,
                           group_name=MODELNUM_NAME,
                           name='learned_models_evaluations')
    logging.info(
        "####################################\n"
        "## Evaluating safety-aware models ##\n"
        "####################################"
    )
    for modelnum in modelnum_iter(learned_path(args.offline_seed)):
        logging.info(f"====== Evaluating model {modelnum} ======")
        env = ContinuousCartPole(
            discretization_shape=shape,  # This matters for the GP
            control_frequency=control_frequency,
            max_theta_init=max_theta_init
        )

        def load_agent(g_opt):
            return DLQRSafetyLearner.load(
                load_path=learned_load_path(args.offline_seed, modelnum),
                env=env,
                x_seed=x_seed,
                y_seed=y_seed,
                gamma_cautious=gamma_cautious,
                lambda_cautious=lambda_cautious,
                gamma_optimistic=g_opt,
                perturbations=perturbations,
                learn_safety=False,
                checks_safety=True,
                is_free_from_safety=False,
            )
        # Try loading the agent with the saved gamma_optimistic
        try:
            agent = load_agent(None)
            logging.info("Loaded agent with saved gamma_measure: "
                         f"{agent.safety_model.gamma_measure}")
        except ValueError:
            agent = load_agent(gamma_optimistic)
            logging.info("Could not load saved gamma_measure. Using "
                         f"{gamma_optimistic} instead")
        t = get_t_from_modelnum(modelnum, n_safety_params_updates)
        agent.update_safety_params(t=t)
        logging.info(f"Updated safety params with t={t}")

        run(agent, args.episodes, args.render, args.log, performances, modelnum)
        del agent
        del env
    logging.info(
        "######################################\n"
        "## Evaluating safety-agnostic model ##\n"
        "######################################"
    )

    env = ContinuousCartPole(
        discretization_shape=shape,  # This matters for the GP
        control_frequency=control_frequency,
        max_theta_init=max_theta_init
    )
    agent = DLQRController(
        env=env,
        perturbations=perturbations
    )
    run(agent, args.episodes, args.render, args.log, performances, -1)

    savepath = offline_path(args.offline_seed) / 'data'
    performances.save(savepath)
