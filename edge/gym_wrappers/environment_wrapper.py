import gym.spaces as gspaces

from edge.envs import Environment
from edge.space import StateActionSpace
from . import BoxWrapper, DiscreteWrapper


class DummyDynamics:
    def __init__(self, stateaction_space):
        self.stateaction_space = stateaction_space

    def is_feasible_state(self, s):
        return True


class GymEnvironmentWrapper(Environment):
    def __init__(self, gym_env, failure_critical = False):
        self.gym_env = gym_env
        if isinstance(gym_env.action_space, gspaces.Box):
             action_space = BoxWrapper(gym_env.action_space)
        elif isinstance(gym_env.action_space, gspaces.Discrete):
            action_space = DiscreteWrapper(gym_env.action_space)
        else:
            raise TypeError(f'Gym environment action_space is of type {type(gym_env.action_space)}, but only Box  '
                             'and Discrete are currently supported')

        if isinstance(gym_env.observation_space, gspaces.Box):
            state_space = BoxWrapper(gym_env.observation_space)
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
        self.gym_env.reset()
        self.s = self.gym_env.obs()
        return self.s

    def step(self, action):
        gym_action = self.action_space.to_gym(action)
        if not self.failure_critical or not self.has_failed:
            gym_new_state, reward, done, info = self.gym_env.step(gym_action)
            self.s = self.state_space.from_gym(gym_new_state)
            self.done = done
            self.info = info
        else:
            reward = 0

        return self.s, reward, self.has_failed