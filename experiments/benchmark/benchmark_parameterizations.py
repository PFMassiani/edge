from numpy import array as nparray
from copy import deepcopy

## Environments and agents names

LOW_GOAL_SLIP = 'low_goal_slip'
PENALIZED_SLIP = 'penalized_slip'
LOW_GOAL_HOVERSHIP = 'low_goal_hovership'
PENALIZED_HOVERSHIP = 'penalized_hovership'

ENVIRONMENTS = [LOW_GOAL_SLIP, PENALIZED_SLIP, LOW_GOAL_HOVERSHIP,
                PENALIZED_HOVERSHIP]

Q_LEARNER = 'q_learner'
SAFETY_Q_LEARNER = 'safety_q_learner'
SOFT_HARD_LEARNER = 'soft_hard_learner'
EPSILON_SAFETY_LEARNER = 'epsilon_safety_learner'
SAFETY_VALUES_SWITCHER = 'safety_values_switcher'

SAFETY_AGENTS = [SOFT_HARD_LEARNER, SAFETY_Q_LEARNER, EPSILON_SAFETY_LEARNER,
                 SAFETY_VALUES_SWITCHER]
NO_SAFETY_AGENTS = [Q_LEARNER]
AGENTS = SAFETY_AGENTS + NO_SAFETY_AGENTS

## Environments parameterizations

LOW_GOAL_SLIP_PARAMS = {
    'reward_done_threshold': None,
    'steps_done_threshold': 10,
    'dynamics_parameters': {'shape': (201, 201)},
}
PENALIZED_SLIP_PARAMS = {
    'penalty_level': 100
}
PENALIZED_SLIP_PARAMS.update(LOW_GOAL_SLIP_PARAMS)

LOW_GOAL_HOVERSHIP_PARAMS = {
    'reward_done_threshold': None,
    'steps_done_threshold': 10,
    'dynamics_parameters': {'shape': (201, 201)},
}
PENALIZED_HOVERSHIP_PARAMS = {
    'penalty_level': 10000
}
PENALIZED_HOVERSHIP_PARAMS.update(LOW_GOAL_HOVERSHIP_PARAMS)

ENVPARAMS_DICT = {
    LOW_GOAL_SLIP: LOW_GOAL_SLIP_PARAMS,
    PENALIZED_SLIP: PENALIZED_SLIP_PARAMS,
    LOW_GOAL_HOVERSHIP: LOW_GOAL_HOVERSHIP_PARAMS,
    PENALIZED_HOVERSHIP: PENALIZED_HOVERSHIP_PARAMS,
}

## Environment-independent agents parameterizations

AGENT_BASE_PARAMS = {
    'greed': 0.1,
    'step_size': 0.6,
    'discount_rate': 0.2,
    # 'discount_rate': 0.99,
    'keep_seed_in_data': True,
}
# Optimistic parameters
SAFETY_AGENT_BASE_PARAMS = {
    'gamma_optimistic': (0.6, 0.8),
    'gamma_cautious': (0.7, 0.8),
    'lambda_cautious': (0., 0.),
}
# Cautious parameters
# SAFETY_AGENT_BASE_PARAMS = {
#     'gamma_optimistic': (0.65, 0.9),
#     'gamma_cautious': (0.7, 0.9),
#     'lambda_cautious': (0.025, 0.025),
# }
Q_LEARNER_PARAMS = AGENT_BASE_PARAMS.copy()
SAFETY_Q_LEARNER_PARAMS = AGENT_BASE_PARAMS.copy()
SAFETY_Q_LEARNER_PARAMS.update(SAFETY_AGENT_BASE_PARAMS)
# Optimistic parameters
SOFT_HARD_LEARNER_PARAMS = {
    'gamma_soft': (0.75, 0.85)
}
# Cautious parameters
# SOFT_HARD_LEARNER_PARAMS = {
#     'gamma_soft': (0.85, 0.95)
# }
SOFT_HARD_LEARNER_PARAMS.update(AGENT_BASE_PARAMS)
SOFT_HARD_LEARNER_PARAMS.update(SAFETY_AGENT_BASE_PARAMS)
EPSILON_SAFETY_PARAMS = {
    'epsilon': 0.1
}
EPSILON_SAFETY_PARAMS.update(AGENT_BASE_PARAMS)
EPSILON_SAFETY_PARAMS.update(SAFETY_AGENT_BASE_PARAMS)
SAFETY_VALUES_SWITCHER_PARAMS = AGENT_BASE_PARAMS.copy()
SAFETY_VALUES_SWITCHER_PARAMS.update(SAFETY_AGENT_BASE_PARAMS)

ENV_INDEP_APARAMS = {
    Q_LEARNER: Q_LEARNER_PARAMS,
    SAFETY_Q_LEARNER: SAFETY_Q_LEARNER_PARAMS,
    SOFT_HARD_LEARNER: SOFT_HARD_LEARNER_PARAMS,
    EPSILON_SAFETY_LEARNER: EPSILON_SAFETY_PARAMS,
    SAFETY_VALUES_SWITCHER: SAFETY_VALUES_SWITCHER_PARAMS,
}

## Environment-dependent agents parameterization updates

SLIP_GP_PARAMS = {
    'outputscale_prior': (0.12, 0.01),
    'lengthscale_prior': (0.15, 0.05),
    'noise_prior': (0.001, 0.002),
    'dataset_type': 'neighborerasing',
    'dataset_params': {'radius': 0.05},
}
SLIP_Q_SEED_PARAMS = {
    'q_x_seed': nparray([.45, 0.6632]),  # .45, 38 / 180 * np.pi
    'q_y_seed': nparray([1]),
}
SLIP_S_SEED_PARAMS = {
    's_x_seed': nparray([[.45, 0.6632], [0.8, 0.4]]),
    's_y_seed': nparray([1, 0.8]),
}

HOVERSHIP_GP_PARAMS = {
    'outputscale_prior': (0.12, 0.01),
    'lengthscale_prior': (0.15, 0.05),
    'noise_prior': (0.001, 0.002),
    'dataset_type': 'neighborerasing',
    'dataset_params': {'radius': 0.05},
}
HOVERSHIP_Q_SEED_PARAMS = {
    'q_x_seed': nparray([[1.3, 0.6], [2, 0]]),
    'q_y_seed': nparray([1, 1]),
}
HOVERSHIP_S_SEED_PARAMS = {
    's_x_seed': nparray([[1.3, 0.6], [2, 0]]),
    's_y_seed': nparray([1, 1]),
}

ENV_DEP_APARAMS = {
    LOW_GOAL_SLIP:
        [SLIP_GP_PARAMS, SLIP_Q_SEED_PARAMS, SLIP_S_SEED_PARAMS],
    PENALIZED_SLIP:
        [SLIP_GP_PARAMS, SLIP_Q_SEED_PARAMS, SLIP_S_SEED_PARAMS],
    LOW_GOAL_HOVERSHIP:
        [HOVERSHIP_GP_PARAMS, HOVERSHIP_Q_SEED_PARAMS, HOVERSHIP_S_SEED_PARAMS],
    PENALIZED_HOVERSHIP:
        [HOVERSHIP_GP_PARAMS, HOVERSHIP_Q_SEED_PARAMS, HOVERSHIP_S_SEED_PARAMS],
}

APARAMS_DICT = {
    aname: {
        envname: deepcopy(ENV_INDEP_APARAMS[aname]) for envname in ENVIRONMENTS
    } for aname in AGENTS
}
for aname in AGENTS:
    for envname in ENVIRONMENTS:
        if aname in SAFETY_AGENTS:
            APARAMS_DICT[aname][envname].update({
                'q_gp_params': ENV_DEP_APARAMS[envname][0],
                's_gp_params': ENV_DEP_APARAMS[envname][0],
            })
            APARAMS_DICT[aname][envname].update(ENV_DEP_APARAMS[envname][1])
            APARAMS_DICT[aname][envname].update(ENV_DEP_APARAMS[envname][2])
        else:
            APARAMS_DICT[aname][envname].update({
                'gp_params': ENV_DEP_APARAMS[envname][0],
            })
            APARAMS_DICT[aname][envname].update(ENV_DEP_APARAMS[envname][1])
            APARAMS_DICT[aname][envname]['x_seed'] = \
                APARAMS_DICT[aname][envname].pop('q_x_seed')
            APARAMS_DICT[aname][envname]['y_seed'] = \
                APARAMS_DICT[aname][envname].pop('q_y_seed')

## Other simulation parameters

SIMULATION_BASE_PARAMS = {
    'n_episodes': 500,
    'glie_start': 0.7,
    'reset_in_safe_state': True,
    'metrics_sampling_frequency': 10,
    'n_episodes_in_measurement': 20,
    'plot_every': 10
}

AGENT_DEP_SIMULATION_PARAMS = {
    Q_LEARNER: {'safety_parameters_update_end': None},
    SAFETY_Q_LEARNER: {'safety_parameters_update_end': None},
    SOFT_HARD_LEARNER: {'safety_parameters_update_end': None},
    EPSILON_SAFETY_LEARNER: {'safety_parameters_update_end': None},
    SAFETY_VALUES_SWITCHER: {'safety_parameters_update_end': 0.4},
}

SIMULATION_PARAMS = AGENT_DEP_SIMULATION_PARAMS.copy()
for aname in AGENTS:
    SIMULATION_PARAMS[aname].update(SIMULATION_BASE_PARAMS)