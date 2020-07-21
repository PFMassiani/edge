import warnings

# The keys of lookup dictionaries should be the name of a parameter in edge, and the values should be their
# corresponding name in vibly
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
    """
    Returns the relevant lookup dictionary for the given environment. This method should be completed as new
    environments are added.
    :param env: the environment
    :return: the relevant lookup dictionary
    """
    # We only do the imports here, otherwise we create an infinite recursive import loop with the edge module: importing
    # any utils from edge requires loading utils.__init__, which itself would require loading edge submodules because
    # of this import.
    from edge.envs import Hovership
    if isinstance(env, Hovership):
        return HOVERSHIP_LOOKUP_DICTIONARY
    elif isinstance(env, Slip):
        return SLIP_LOOKUP_DICTIONARY
    else:
        warnings.warn("Seems like you haven't implemented some " +
                      "compabibility stuff for your environment yet! See " +
                      "`utils/vibly_compatibility_utils.py`")
        return None
