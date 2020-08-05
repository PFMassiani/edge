import gym.spaces as gspaces

from edge.envs import Environment
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
    def __init__(self, gym_env, shape=None, failure_critical = False):
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

        dynamics = DummyDynamics(StateActionSpace(state_space, action_space))

        super(GymEnvironmentWrapper, self).__init__(
            dynamics=dynamics,
            reward=None,
            default_initial_state=gym_env.reset(),
            random_start=True
        )

        self.info = {}
        self.done = False
        self.failure_critical = False

    @property
    def in_failure_state(self):
        cost = self.info.get('cost')
        return cost is not None and cost != 0

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
        return self.s

    def step(self, action):
        gym_action = self.action_space.to_gym(action)
        if not self.failure_critical or not self.has_failed:
            gym_new_state, reward, done, info = self.gym_env.step(gym_action)
            s = self.state_space.from_gym(gym_new_state)
            # Gym does not put a hard constraint on the fact that the state
            # stays in the limit of the Box. Edge crashes if this happens, so
            # we project the resulting state in state-space
            self.s = self.state_space.closest_in(s)
            self.done = done
            self.info = info
        else:
            reward = 0

        return self.s, reward, self.has_failed

    def render(self):
        self.gym_env.render()