import functools
import time
import pandas as pd
import json
import logging

from edge.dataset import Dataset
from edge.utils.logging import config_msg

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


def avg_reward_and_failure(df, group_key):
    r = df.groupby(group_key)[Dataset.REWARD].sum().mean()
    f = df.groupby(group_key)[Dataset.FAILED].any().mean()
    return r, f


def average_performances(df, group_name, episode_name, last_n_episodes=None):
    group_key = list((group_name, episode_name))
    ep_global = pd.Series(df.groupby(group_key).grouper.group_info[0],
                          index=df.index)
    # ep_global = (ep_change.diff() != 0).fillna(False).cumsum()
    ep_max = ep_global.max()
    if last_n_episodes is None:
        last_n_episodes = ep_max + 1
    r, f = avg_reward_and_failure(
        df.loc[ep_global > ep_max - last_n_episodes],
        group_key
    )
    return r, f


def affine_interpolation(t, start, end):
    return start + (end - start) * t


def log_simulation_parameters(f):
    @functools.wraps(f)
    def logged_function(*args, **kwargs):
        out = f(*args, **kwargs)
        message = (f'Function {f.__name__} called with parameters:\n'
                   f'ARGS:\n{args}\nKWARGS:{json.dumps(kwargs, indent=1)}')
        logging.info(config_msg(message))
    return logged_function
