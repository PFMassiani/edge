import argparse
from pathlib import Path
import numpy as np
import logging
import gpytorch

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
SAFE_RESET = 'Safe reset'
SAMPLE_PRIOR = True
CHECK_VIAB = True


def average_performances(ds):
    df = ds.df
    r, f = general_performances(
        df, ds.group_name, ds.EPISODE, None
    )
    safe_reset_df = df.loc[df[SAFE_RESET]]
    safe_r, safe_f = general_performances(
        safe_reset_df, ds.group_name, ds.EPISODE, None
    )
    # s = df.groupby(ds.group_name)[SAFETY_NAME].mean().mean()
    return r, f, safe_r, safe_f


def append_to_episode(dataset, episode, state, action, new_state, reward,
                      failed, done, safe_reset):
    general_append(dataset, episode, state, action, new_state, reward,
                   failed, done)
    episode[SAFE_RESET].append(safe_reset)

def reset(agent, check_initial_viability=False, safety_measure=None,
          gamma_prior=0.6):
    t = 0
    reset_done = False
    while not reset_done:
        t += 1
        agent.reset()
        if check_initial_viability:
            gamma = safety_measure.gamma_measure if not SAMPLE_PRIOR else\
                gamma_prior
            with gpytorch.settings.prior_mode(state=SAMPLE_PRIOR):
                measure = safety_measure.measure(
                    state=agent.state,
                    lambda_threshold=0,
                    gamma_threshold=gamma
                )
            reset_done = (not agent.env.done) and (measure[0] > 0)
        else:
            reset_done = not agent.env.done
        if t >= 100 and check_initial_viability:
            msg = (f"Could not reset the agent after {t} attempts with " 
                   "check_initial_viability enabled. Falling back on default "
                   "agent.reset() behaviour")
            logging.warning(msg)
            reset(agent)
            break
    return reset_done


def run_episode(agent, ds, render, check_initial_viability=False,
                safety_measure=None, gamma_prior=0.6):
    with gpytorch.settings.prior_mode(state=SAMPLE_PRIOR):
        episode = {cname: []
                   for cname in ds.columns_wo_group}
        reset_successful = reset(agent, check_initial_viability, safety_measure,
                                 gamma_prior)
        done = agent.env.done
        while not done:
            old_state = agent.state
            new_state, reward, failed, done = agent.step()
            action = agent.last_action
            # try:
            #     safety_update = agent.safety_update if agent.do_safety_update \
            #         else None
            # except AttributeError:
            #     safety_update = None
            append_to_episode(ds, episode, old_state, action,
                              new_state, reward, failed, done, reset_successful)
            if render:
                agent.env.gym_env.render()
    return episode


def run(agent, max_episodes, render, log_every, performances, modelnum,
        check_initial_viability, safety_measure, gamma_prior):
    # ds = Dataset(*Dataset.DEFAULT_COLUMNS, SAFETY_NAME)
    # performances = Dataset(Dataset.REWARD, Dataset.FAILED, name=name)
    ds = Dataset(*Dataset.DEFAULT_COLUMNS, SAFE_RESET)
    for n_episode in range(max_episodes):
        episode = run_episode(agent, ds, render, check_initial_viability,
                              safety_measure, gamma_prior)
        ds.add_group(episode, group_number=n_episode)
        if (n_episode + 1) % log_every == 0:
            r, f, s_r, s_f = log_performance(ds, n_episode+1, max_episodes)
            can_safe_reset = ds.df[SAFE_RESET].any()
            performances.add_entry(modelnum, n_episode, r, f, s_r, s_f,
                                   can_safe_reset)
    if render:
        agent.env.gym_env.close()


def log_performance(ds, n_episode, max_episodes):
    r, f, safe_r, safe_f = average_performances(ds)
    if np.isnan(safe_r):
        safe_r = 0.
    if np.isnan(safe_f):
        safe_f = 1.
    header = f'------ Checkpoint {n_episode}/{max_episodes} ------\n'
    message = (f'Average total reward per episode: {r:.3f} (all) '
               f'| {safe_r:.3f} (safe resets)\n'
               f'Average number of failures: {f * 100:.3f} % (all) '
               f'| {safe_f * 100:.3f} % (safe resets)\n')
               # f'Average next state safety: {s:.3f} %\n')
    logging.info(header + message)
    return r, f, safe_r, safe_f


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
    parser.add_argument('--episodes', default=50)
    parser.add_argument('--render', default=False)
    parser.add_argument('--log', default=1)

    args = parser.parse_args()

    suffix = '_safe_reset' if CHECK_VIAB else ''
    logname = ('prior_' if SAMPLE_PRIOR else '') + f'evaluation{suffix}.log'
    setup_default_logging_configuration(
        log_path=offline_path(args.offline_seed) / 'logs' / logname
    )

    n_safety_params_updates = 100
    gamma_cautious = (0.6, 0.9)
    lambda_cautious = (0., 0.05)
    gamma_optimistic = (0.55, 0.85)
    max_theta_init = 0.4
    shape = (50, 50, 50, 50, 41)
    perturbations = {'g': 1/1, 'mcart': 1, 'mpole': 1, 'l': 1}
    control_frequency = 2
    x_seed = np.array([[0, 0, 0, 0, 0.]])
    y_seed = np.array([1.])


    def load_agent(g_opt, modelnum):
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

    name = 'learned_models_evaluations' if not SAMPLE_PRIOR else \
        'priors_evaluations'
    if CHECK_VIAB:
        name += suffix
    suffix = ' (safe reset)'
    performances = Dataset(Dataset.EPISODE, Dataset.REWARD, Dataset.FAILED,
                           Dataset.REWARD + suffix, Dataset.FAILED + suffix,
                           SAFE_RESET,
                           group_name=MODELNUM_NAME, name=name)
    logging.info(
        "####################################\n"
        "## Evaluating safety-aware models ##\n"
        "####################################"
    )
    best_modelnum = None
    best_perf = None
    for modelnum in modelnum_iter(learned_path(args.offline_seed)):
        logging.info(f"====== Evaluating model {modelnum} ======")
        env = ContinuousCartPole(
            discretization_shape=shape,  # This matters for the GP
            control_frequency=control_frequency,
            max_theta_init=max_theta_init
        )

        # Try loading the agent with the saved gamma_optimistic
        try:
            agent = load_agent(None, modelnum)
            logging.info("Loaded agent with saved gamma_measure: "
                         f"{agent.safety_model.gamma_measure}")
        except ValueError:
            agent = load_agent(gamma_optimistic, modelnum)
            logging.info("Could not load saved gamma_measure. Using "
                         f"{gamma_optimistic} instead")
        t = get_t_from_modelnum(modelnum, n_safety_params_updates)
        agent.update_safety_params(t=t)
        logging.info(f"Updated safety params with t={t}")

        run(agent, args.episodes, args.render, args.log, performances, modelnum,
            CHECK_VIAB, agent.safety_model, gamma_cautious[0])
        # try:
        model_perfs = performances.df.loc[
            performances.df[MODELNUM_NAME] == modelnum, :
        ]
        agent_perf = model_perfs.loc[model_perfs.index[-1], Dataset.REWARD]
        can_initialize_viably = model_perfs[SAFE_RESET].any()
        # except IndexError:
        #     agent_perf = 0.

        if best_modelnum is None or (
                can_initialize_viably and
                (best_perf < agent_perf)
        ):
            best_modelnum = modelnum
            best_perf = agent_perf
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
    if CHECK_VIAB:
        try:
            best_agent = load_agent(None, best_modelnum)
        except ValueError:
            best_agent = load_agent(gamma_optimistic, best_modelnum)
        t = get_t_from_modelnum(best_modelnum, n_safety_params_updates)
        best_agent.update_safety_params(t=t)
        best_measure = best_agent.safety_model
    else:
        best_measure = None
    run(agent, args.episodes, args.render, args.log, performances, -1,
        CHECK_VIAB, best_measure, gamma_cautious[0])

    savepath = offline_path(args.offline_seed) / 'data'
    performances.save(savepath)
