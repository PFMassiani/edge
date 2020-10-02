import gym
import numpy as np

from edge.space import StateActionSpace, ProductSpace
from edge.gym_wrappers import GymEnvironmentWrapper


class LunarLander(GymEnvironmentWrapper):
    def __init__(self, shape, control_frequency):
        gym_env = gym.make('LunarLander-v2')

        # We remove the last two components of the state space
        self._state_mask = slice(None, -2, None)

        super(LunarLander, self).__init__(
            gym_env=gym_env,
            shape=shape,
            failure_critical=True,
            control_frequency=control_frequency,
            inf_ceiling=10.,  # From Gym, failure happens at abs(s[0] > 1)
        )
        state_space, action_space = self.dynamics.stateaction_space.sets
        state_space_components = state_space._flattened_sets
        state_space = ProductSpace(*state_space_components[self._state_mask])
        self._stateaction_space = StateActionSpace(state_space, action_space)

    @GymEnvironmentWrapper.stateaction_space.getter
    def stateaction_space(self):
        return self._stateaction_space

    @GymEnvironmentWrapper.s.setter
    def s(self, new_s):
        if len(new_s) == 6:
            self._s = new_s
        else:
            self._s = new_s[self._state_mask]

    @property
    def in_failure_state(self):
        # Condition taken from the code of OpenAI Gym
        return self.gym_env.game_over or abs(self.s[0]) >= 1.0


class PenalizedLunarLander(LunarLander):
    def __init__(self, penalty_level, shape, control_frequency):
        super(PenalizedLunarLander, self).__init__(shape, control_frequency)
        self.penalty_level = penalty_level

    def step(self, action):
        s, reward, failed = super(LunarLander, self).step(action)
        if self.in_failure_state:
            reward -= self.penalty_level

        return s, reward, failed