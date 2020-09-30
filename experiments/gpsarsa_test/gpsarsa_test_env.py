import gym

from edge.envs import Hovership
from edge.gym_wrappers import GymEnvironmentWrapper
from edge.reward import AffineReward, ConstantReward


class LowGoalHovership(Hovership):
    def __init__(self, dynamics_parameters=None, steps_done_threshold=None):
        super(LowGoalHovership, self).__init__(
            random_start=True,
            dynamics_parameters=dynamics_parameters,
            steps_done_threshold=steps_done_threshold,
        )

        reward = AffineReward(self.stateaction_space, [(10, 0), (0, 0)])
        self.reward = reward


class PenalizedHovership(LowGoalHovership):
    def __init__(self, penalty_level=100, dynamics_parameters=None,
                 steps_done_threshold=None):
        super(PenalizedHovership, self).__init__(dynamics_parameters,
                                                 steps_done_threshold)

        def penalty_condition(state, action, new_state, reward):
            return self.is_failure_state(new_state)

        penalty = ConstantReward(self.reward.stateaction_space, -penalty_level,
                                 reward_condition=penalty_condition)

        self.reward += penalty


class HighGoalHovership(Hovership):
    def __init__(self, dynamics_parameters=None, steps_done_threshold=None):
        super(HighGoalHovership, self).__init__(
            random_start=True,
            dynamics_parameters=dynamics_parameters,
            steps_done_threshold=steps_done_threshold,
        )

        reward = AffineReward(self.stateaction_space, [(0, 10), (0, 0)])
        self.reward = reward


class HighPenalizedHovership(HighGoalHovership):
    def __init__(self, penalty_level=100, dynamics_parameters=None,
                 steps_done_threshold=None):
        super(HighPenalizedHovership, self).__init__(dynamics_parameters,
                                                     steps_done_threshold)

        def penalty_condition(state, action, new_state, reward):
            return self.is_failure_state(new_state)

        penalty = ConstantReward(self.reward.stateaction_space, -penalty_level,
                                 reward_condition=penalty_condition)

        self.reward += penalty


class LunarLander(GymEnvironmentWrapper):
    def __init__(self, shape, control_frequency):
        gym_env = gym.make('LunarLander-v2')
        super(LunarLander, self).__init__(
            gym_env=gym_env,
            shape=shape,
            failure_critical=True,
            control_frequency=control_frequency,
            inf_ceiling=10.,  # From Gym, failure happens at abs(s[0] > 1)
        )

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