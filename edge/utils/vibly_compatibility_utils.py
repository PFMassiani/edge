HOVERSHIP_LOOKUP_DICTIONARY = {
    'base_gravity': 'ground_gravity',
    'gravity': 'gravity_gradient',
    'control_frequency': 'control_frequency',
    'max_thrust': 'max_thrust',
    'ceiling': 'max_altitude',
    'n_states': None,
    'thrust': None
}


def get_parameters_lookup_dictionary(env):
    from edge.envs import Hovership
    if isinstance(env, Hovership):
        return HOVERSHIP_LOOKUP_DICTIONARY
    else:
        return None
