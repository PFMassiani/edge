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

        self._info = {}
        self.done = False
        self.failure_critical = False

    @property
    def in_failure_state(self):
        cost = self._info.get('cost')
        return cost is not None and cost != 0

    def step(self, action):
        gym_action = self.action_space.to_gym(action)
        if self.failure_critical and not self.has_failed:
            gym_new_state, reward, done, info = self.gym_env.step(gym_action)
            self.s = self.state_space.from_gym(gym_new_state)
            self.done = done
            self._info = info
        else:
            reward = 0

        return self.s, reward, self.has_failed