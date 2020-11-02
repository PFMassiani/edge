from edge.model.policy_models import Policy
from edge.utils.control import dlqr


class DLQRPolicy(Policy):
    def __init__(self, stateaction_space, A, B, Q, R):
        super().__init__(stateaction_space)
        self.Q = Q
        self.R = R
        self.K, _, _ = dlqr(A, B, Q, R)

    def get_action(self, state):
        action = - self.K @ state
        return self.stateaction_space.action_space.closest_in(action)
