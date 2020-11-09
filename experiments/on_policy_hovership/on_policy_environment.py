from edge.envs import Hovership, Slip
from edge.reward import AffineReward


class LowGoalHovership(Hovership):
    def __init__(self, shape=(201, 161), reward_done_threshold=None,
                 steps_done_threshold=10, goal_state=False,
                 initial_state=None):
        super(LowGoalHovership, self).__init__(
            random_start=initial_state is None,
            default_initial_state=initial_state,
            dynamics_parameters={'shape': shape},
            reward_done_threshold=reward_done_threshold,
            steps_done_threshold=steps_done_threshold,
            goal_state=goal_state,
        )

        reward = AffineReward(self.stateaction_space, [(11, 1), (1, 0)])
        self.reward = reward


class LowGoalSlip(Slip):
    # * This goal incentivizes the agent to run fast
    def __init__(self, shape=(200, 161), reward_done_threshold=200,
                 steps_done_threshold=None, initial_state=None):
        super(LowGoalSlip, self).__init__(
            dynamics_parameters={'shape': shape},
            reward_done_threshold=reward_done_threshold,
            steps_done_threshold=steps_done_threshold,
            random_start=initial_state is None,
            default_initial_state=initial_state,
        )

        reward = AffineReward(self.stateaction_space, [(10, 0), (0, 0)])
        self.reward = reward
