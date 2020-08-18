from edge.envs import Slip, Hovership
from edge.reward import AffineReward, ConstantReward


class LowGoalSlip(Slip):
    # * This goal incentivizes the agent to run fast
    def __init__(self, dynamics_parameters=None, reward_done_threshold=None):
        super(LowGoalSlip, self).__init__(
            dynamics_parameters=dynamics_parameters,
            reward_done_threshold=reward_done_threshold,
            random_start=True
        )

        reward = AffineReward(self.stateaction_space, [(1, 0), (0, 0)])
        self.reward = reward


class PenalizedSlip(LowGoalSlip):
    def __init__(self, penalty_level=100, dynamics_parameters=None,
                 reward_done_threshold=None):
        super(PenalizedSlip, self).__init__(dynamics_parameters,
                                            reward_done_threshold)

        def penalty_condition(state, action, new_state, reward):
            return self.is_failure_state(new_state)

        penalty = ConstantReward(self.reward.stateaction_space, -penalty_level,
                                 reward_condition=penalty_condition)

        self.reward += penalty


class LowGoalHovership(Hovership):
    def __init__(self, dynamics_parameters=None, reward_done_threshold=None):
        super(LowGoalHovership, self).__init__(
            random_start=True,
            dynamics_parameters=dynamics_parameters,
            reward_done_threshold=reward_done_threshold,
        )

        reward = AffineReward(self.stateaction_space, [(10, 0), (0, 0)])
        self.reward = reward


class PenalizedHovership(LowGoalHovership):
    def __init__(self, penalty_level=100, dynamics_parameters=None,
                 reward_done_threshold=None):
        super(PenalizedHovership, self).__init__(dynamics_parameters,
                                                 reward_done_threshold)

        def penalty_condition(state, action, new_state, reward):
            return self.is_failure_state(new_state)

        penalty = ConstantReward(self.reward.stateaction_space, -penalty_level,
                                 reward_condition=penalty_condition)

        self.reward += penalty