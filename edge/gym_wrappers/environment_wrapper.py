import gym.spaces as gspaces

from edge.envs.environments import Environment
from edge.space import StateActionSpace
from . import BoxWrapper, DiscreteWrapper


class DummyDynamics:
    def __init__(self, stateaction_space):
        self.stateaction_space = stateaction_space

    def is_feasible_state(self, s):
        return True


def get_index_length(gym_box):
    return len(gym_box.low.reshape(-1))


class GymEnvironmentWrapper(Environment):
    def __init__(self, gym_env, shape=None, failure_critical=False,
                 control_frequency=None):
        self.gym_env = gym_env
        if shape is None:
            obs_shape = None
            action_shape = None
        if isinstance(gym_env.action_space, gspaces.Box):
            if shape is not None:
                action_space_ndim = get_index_length(gym_env.action_space)
                action_shape = shape[-action_space_ndim:]
            action_space = BoxWrapper(
                gym_env.action_space, discretization_shape=action_shape
            )
        elif isinstance(gym_env.action_space, gspaces.Discrete):
            action_space = DiscreteWrapper(gym_env.action_space)
        else:
            raise TypeError(f'Gym environment action_space is of type {type(gym_env.action_space)}, but only Box  '
                             'and Discrete are currently supported')

        if isinstance(gym_env.observation_space, gspaces.Box):
            if shape is not None:
                state_space_ndim = get_index_length(gym_env.observation_space)
                obs_shape = shape[:state_space_ndim]
            state_space = BoxWrapper(
                gym_env.observation_space,
                discretization_shape=obs_shape
            )
        elif isinstance(gym_env.observation_space, gspaces.Discrete):
            state_space = DiscreteWrapper(gym_env.observation_space)
        else:
            raise TypeError(f'Gym environment observation_space is of type {type(gym_env.observation_space)}, but only '
                            'Box and Discrete are currently supported')

        self.info = {}
        self._done = False
        self.failure_critical = failure_critical
        self.control_frequency = control_frequency

        dynamics = DummyDynamics(StateActionSpace(state_space, action_space))

        super(GymEnvironmentWrapper, self).__init__(
            dynamics=dynamics,
            reward=None,
            default_initial_state=gym_env.reset(),
            random_start=True
        )

    @property
    def in_failure_state(self):
        cost = self.info.get('cost')
        return cost is not None and cost != 0

    @property
    def done(self):
        return self._done

    @property
    def has_failed(self):
        # Same as in_failure_state, for compatibility with generic Environment
        return self.in_failure_state

    def reset(self, s=None):
        # Usually, Environments take s as a parameter, but this is not supported by safety_gym, so we
        # raise a meaningful error for the user
        if s is not None:
            raise ValueError('Selecting the initial state is not supported for Gym environments')
        reset_output = self.gym_env.reset()
        # Safety gym does not return anything with reset, whereas Gym returns
        # the state
        if reset_output is None:
            self.s = self.gym_env.obs()
        else:
            self.s = reset_output
        self._done = self.in_failure_state
        return self.s

    def step(self, action):
        gym_action = self.action_space.to_gym(action)

        def do_one_gym_step():
            if not self.failure_critical or not self.has_failed:
                gym_new_state, reward, done, info = self.gym_env.step(
                    gym_action
                )
                s = self.state_space.from_gym(gym_new_state)
                # Gym does not put a hard constraint on the fact that the state
                # stays in the limit of the Box. Edge crashes if this happens,
                # so we project the resulting state in state-space
                s = self.state_space.closest_in(s)
            else:
                reward = 0
            return s, reward, done, info

        step_done = False
        n_gym_steps = 0
        while not step_done:
            self.s, reward, self._done, _ = do_one_gym_step()
            n_gym_steps += 1
            step_done = (self.control_frequency is None) or \
                        (n_gym_steps >= self.control_frequency) or \
                        (self._done)

        return self.s, reward, self.has_failed

    def render(self):
        self.gym_env.render()

    def compute_dynamics_map(self):
        # General note: Q_map stores the index of the next state. This
        # approximates the dynamics by projecting the state we end up in, and
        # may lead to errors. A more precise implementation would keep the
        # exact value of the next state instead of its index. So far, this
        # method is only used for the computation of the viability sets, and
        # this requires the index of the next state: implementing the more
        # precise method is useless for this.
        # However, the following implementation may need to change if this
        # method is used for something else.

        import numpy as np
        unwrapped_gym_env = self.gym_env.unwrapped
        Q_map = np.zeros(self.stateaction_space.shape, dtype=tuple)
        for sa_index, stateaction in iter(self.stateaction_space):
            state, action = self.stateaction_space.get_tuple(stateaction)
            state = self.state_space.to_gym(state)
            action = self.action_space.to_gym(action)
            unwrapped_gym_env.state = state
            next_state, reward, failed = self.step(action)
            next_state = self.state_space.from_gym(next_state)
            # Gym does not ensure the stability of the stateaction space under
            # the dynamics, so we enforce it.
            # This may lead to edge effects.
            next_state = self.state_space.closest_in(next_state)
            next_state_index = self.state_space.get_index_of(
                next_state, around_ok=True
            )
            Q_map[sa_index] = next_state_index
        return Q_map