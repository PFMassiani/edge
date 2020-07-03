import warnings

HOVERSHIP_LOOKUP_DICTIONARY = {
    'base_gravity': 'ground_gravity',
    'gravity': 'gravity_gradient',
    'control_frequency': 'control_frequency',
    'max_thrust': 'max_thrust',
    'ceiling': 'max_altitude',
    'n_states': None,
    'thrust': None
}

SLIP_LOOKUP_DICTIONARY = {
    'gravity': 'gravity',
    'mass': 'mass',
    'stiffness': 'stiffness',
    'resting_length': 'resting_length',
    'total_energy': 'energy',
    'angle_of_attack': None,
    'actuator_resting_length': None,
    'x0': None
}


def get_parameters_lookup_dictionary(env):
    from edge.envs import Hovership, Slip
    if isinstance(env, Hovership):
        return HOVERSHIP_LOOKUP_DICTIONARY
    elif isinstance(env, Slip):
        return SLIP_LOOKUP_DICTIONARY
    else:
        warnings.warn("Seems like you haven't implemented some " +
                      "compabibility stuff for your environment yet! See " +
                      "`utils/vibly_compatibility_utils.py`")
        return None
