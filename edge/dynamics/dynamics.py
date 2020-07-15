import numpy as np

from .event import EventBased
from edge import error


class DiscreteTimeDynamics(EventBased):
    """Represents discrete time dynamics.
    The 'step' method computes the state at the next time.
    """
    def __init__(self, stateaction_space):
        """Initializer.
        Subclasses should initialize the stateaction_space object. The given space should be the one that will be
        accessed externally: Dynamics can used a richer state-action space internally, but this richer space should
        not be this argument.
        :param stateaction_space: edge.space.StateActionSpace. The externally-accessed stateaction_space
        """
        self.stateaction_space = stateaction_space

    @property
    def parameters(self):
        """ Abstract property
        Returns the dict of the parameters relevant for the dynamics
        :return: dictionary of parameters
        """
        raise NotImplementedError
        return {}

    def step(self, state, action):
        """ Abstract method
        Computes the next state
        :param state: np.ndarray. The current state
        :param action: np.ndarray. The action taken
        :return: np.ndarray. The next state
        """
        raise NotImplementedError

    def is_feasible_state(self, state):
        """Unused, returns True"""
        return True

    def compute_map(self):
        """Computes the dynamics map
        :return: np.ndarray. The dynamics map. The array has n_s + n_a dimensions, and the values are the indexes in
            self.stateaction_space
        """
        # This probably breaks when self.stateaction_space.state_space.ndim > 1: then, indexes are tuples, not ints
        Q_map = np.zeros(self.stateaction_space.shape, dtype=int)
        for sa_index, stateaction in iter(self.stateaction_space):
            state, action = self.stateaction_space.get_tuple(stateaction)
            next_state, failed = self.step(state, action)
            next_state_index = self.stateaction_space.state_space.get_index_of(
                next_state
            )
            Q_map[sa_index] = next_state_index
        return Q_map



class TimestepIntegratedDynamics(DiscreteTimeDynamics):
    """DiscreteTimeDynamics where the time is discretized, but the dynamics between two timesteps are simulated
    using a continuous model (e.g., solving an ODE).
    """
    def __init__(self, stateaction_space):
        """
        :param stateaction_space: edge.space.StateActionSpace. The externally-accessed stateaction_space
        """
        super(TimestepIntegratedDynamics, self).__init__(stateaction_space)

    def get_trajectory(self, state, action):
        """ Abstract method
        Used to encapsulate the simulation of the trajectory from `step`. If you decide not to redefine this
        method, you should redefine `step`.
        :param state: The starting state
        :param action: The action taken
        :return: output of scipy.integrate.solve_ivp
        """
        raise NotImplementedError

    def ensure_in_state_space(self, new_state):
        """ Projects the argument onto the stateaction_space
        `get_trajectory` may compute "states" that are outside of the state space (due to thresholding effects, for
        example). This method solves that problem
        :param new_state: The state you want to project
        :return: An element of self.stateaction_space
        """
        if new_state in self.stateaction_space.state_space:
            return new_state
        else:
            return self.stateaction_space.state_space.closest_in(new_state)

    def step(self, state, action):
        """ Main method
        Computes the next state. If you want to use a rich stateaction_space to take a step, you probably should
        redefine this method.
        :param state: np.ndarray. The current state
        :param action: np.ndarray. The action taken
        :return: np.ndarray. The next state
        """
        if (state not in self.stateaction_space.state_space) or (
                action not in self.stateaction_space.action_space):
            raise error.OutOfSpace
        if not self.is_feasible_state(state):
            return state, False

        trajectory = self.get_trajectory(state, action)
        new_state = np.atleast_1d(trajectory.y[:, -1])
        new_state = self.ensure_in_state_space(new_state)
        is_feasible = self.is_feasible_state(new_state)

        return new_state, is_feasible
