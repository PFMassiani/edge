from edge.agent import Agent
from edge.utils.control import dlqr


class DLQRController(Agent):
    def __init__(self, env, Q, R):
        super(DLQRController, self).__init__(env)
        self.Q = Q
        self.R = R
        A, B = env.linearization()
        self.K, _, _ = dlqr(A, B, Q, R)

    def get_next_action(self):
        action = - self.K @ self.state
        return self.env.action_space.closest_in(action)

    def step(self):
        """
        Chooses an action according to the policy, takes a step in the Environment, and updates the models. The action
        taken is available in self.last_action.
        :return: new_state, reward, failed
        """
        action = self.get_next_action()
        new_state, reward, failed = self.env.step(action)
        self.state = new_state
        self.last_action = action
        return new_state, reward, failed, self.env.done