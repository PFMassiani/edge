import argparse
from pathlib import Path
import numpy as np
from on_policy_agent import RandomSafetyLearner, AffineSafetyLearner
from on_policy_environment import LowGoalHovership
from edge.model.safety_models import SafetyTruth
from edge.dataset import Dataset

AFFINE = 'affine'
RANDOM = 'random'


def learned_qv(agent, safety_truth, cautious=False):
    Q_learned = agent.safety_model.level_set(
        state=None,  # Whole state-space
        lambda_threshold=0,
        gamma_threshold=agent.gamma_optimistic if not cautious else agent.gamma_cautious
    ).astype(bool)
    Q_V = safety_truth.viable_set_like(
        agent.env.stateaction_space
    ).astype(bool)
    learned_qv_ratio = (Q_V & Q_learned).astype(int).sum()
    learned_qv_ratio /= Q_V.astype(int).sum()
    return learned_qv_ratio


def closest_viable_action(state, action, safety_truth, env):
    d = np.inf
    closest_a = None
    for _, a in env.stateaction_space.action_space:
        current_d = np.linalg.norm(a - action)
        if (current_d < d) and safety_truth.is_viable(state, a):
            d = current_d
            closest_a = a
    if closest_a is None:
        closest_a = action
    return closest_a


def difference(agent, safety_truth):
    m = 0
    inf = 0
    n = 0
    for _, s in iter(agent.env.stateaction_space.state_space):
        is_viable = safety_truth.is_viable(state=s)
        if is_viable:
            policy_action = agent.policy.get_action(s).flatten()
            optimal_action = closest_viable_action(s, policy_action,
                                                   safety_truth, agent.env)
            agent.state = s
            agent_action = agent.get_next_action().flatten()
            d = np.linalg.norm(optimal_action - agent_action)
            m = d / (n+1) + m * n / (n+1)
            n += 1
            inf = max(inf, d)
    return m, inf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nominal', default=RANDOM)
    parser.add_argument('--nmodel', default=1, type=int)
    args = parser.parse_args()

    gamma_cautious = (0.75, 0.75)
    lambda_cautious = (0, 0.0)

    here = Path(__file__).absolute().parent
    apath = here / f'{args.nominal}_controller' / 'models' / 'safety_model' / \
            f'safety_model_{args.nmodel}'

    env = LowGoalHovership(
        goal_state=False,
        initial_state=np.array([1.3]),
    )

    if args.nominal == AFFINE:
        agent = AffineSafetyLearner.load(
            env=env,
            mpath=apath,
            offset=(np.array([2.0]), np.array([0.1])),
            jacobian=np.array([[(0.7 - 0.1) / (0. - 2.)]]),
            gamma_cautious=gamma_cautious,
            lambda_cautious=lambda_cautious
        )
    elif args.nominal == RANDOM:
        agent = RandomSafetyLearner.load(
            env=env,
            mpath=apath,
            gamma_cautious=gamma_cautious,
            lambda_cautious=lambda_cautious
        )
    else:
        raise ValueError

    truth_path = here.parent.parent / 'data' / 'ground_truth' / 'from_vibly' / \
                 'hover_map.pickle'
    ground_truth = SafetyTruth(env)
    ground_truth.from_vibly_file(truth_path)

    dataset_path = here / f'{args.nominal}_controller' / 'data' / 'train.csv'
    dataset = Dataset.load(dataset_path, group_name='Training')

    print(f"EVALUATING {args.nominal} AGENT AFTER BATCH #{args.nmodel}")
    n_samples = len(dataset.loc[dataset.df['Training'] <= args.nmodel])
    print(f'Number of training samples: {n_samples}')
    optimistic_qv_ratio = learned_qv(agent, ground_truth, cautious=False)
    print(f"Q_opt / Q_V ratio: {optimistic_qv_ratio*100:.3f} %")
    cautious_qv_ratio = learned_qv(agent, ground_truth, cautious=True)
    print(f"Q_caut / Q_V ratio: {cautious_qv_ratio*100:.3f} %")
    if args.nominal == AFFINE:
        mean_diff, inf_diff = difference(agent, ground_truth)
        print(f"L2 difference with optimal controller (state average): "
              f"{mean_diff:.3f}")
        print(f"L_inf difference with optimal controller: "
              f"{inf_diff:.3f}")


