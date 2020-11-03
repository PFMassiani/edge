import argparse
from pathlib import Path
import numpy as np

from edge.envs.continuous_cartpole import ContinuousCartPole
from edge.dataset import Dataset
from edge.utils import append_to_episode as general_append, \
    average_performances as general_performances
from edge.utils import append_to_episode, average_performances

from learned_mean_agent import DLQRSafetyLearner

SAFETY_NAME = 'Next safety'


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
        safety_update = agent.safety_update if agent.do_safety_update else None
        append_to_episode(ds, episode, old_state, action,
                          new_state, reward, failed, done, safety_update)
        if render:
            agent.env.gym_env.render()
    return episode


def run(agent, max_episodes, render, log_every):
    # ds = Dataset(*Dataset.DEFAULT_COLUMNS, SAFETY_NAME)
    ds = Dataset()
    for n_episode in range(max_episodes):
        episode = run_episode(agent, ds, render)
        ds.add_group(episode, group_number=n_episode)
        if (n_episode + 1) % log_every == 0:
            log_performance(ds, n_episode+1, max_episodes)
    if render:
        agent.env.gym_env.close()


def log_performance(ds, n_episode, max_episodes):
    r, f = average_performances(ds)
    header = f'-------- {n_episode}/{max_episodes} --------\n'
    message = (f'Average total reward per episode: {r:.3f}\n'
               f'Average number of failures: {f * 100:.3f} %\n')
               # f'Average next state safety: {s:.3f} %\n')
    print(header + message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('simname')
    parser.add_argument('modelnum')
    parser.add_argument('--episodes', default=6)
    parser.add_argument('--render', default=True)
    parser.add_argument('--log', default=2)

    args = parser.parse_args()

    simpath = Path(__file__).parent.resolve() / args.simname / 'models' / \
        'safety_model' / f'safety_model_{args.modelnum}'

    gamma_cautious = 0.7
    lambda_cautious = 0.05
    gamma_optimistic = 0.65
    max_theta_init = 0.4
    shape = (50, 50, 50, 50, 41)
    perturbations = {'g': 1/1, 'mcart': 1, 'mpole': 1, 'l': 1}
    control_frequency = 2
    x_seed = np.array([[0, 0, 0, 0, 0.]])
    y_seed = np.array([1.])

    env = ContinuousCartPole(
        discretization_shape=shape,  # This matters for the GP
        control_frequency=control_frequency,
        max_theta_init=max_theta_init
    )

    agent = DLQRSafetyLearner.load(
        load_path=simpath,
        env=env,
        x_seed=x_seed,
        y_seed=y_seed,
        gamma_cautious=gamma_cautious,
        lambda_cautious=lambda_cautious,
        gamma_optimistic=gamma_optimistic,
        perturbations=perturbations,
        learn_safety=True,
        checks_safety=False
    )

    run(agent, args.episodes, args.render, args.log)
