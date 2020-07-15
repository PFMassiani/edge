from . import Policy
from edge.gym_wrappers import GymEnvironmentWrapper


class MLPPolicy(Policy):
    def __init__(self, env, perceptron):
        super(MLPPolicy, self).__init__(env.stateaction_space)
        self.gym_wrapping = isinstance(env, GymEnvironmentWrapper)
        self.perceptron = perceptron
        self.env = env

    def get_action(self, state):
        action = self.perceptron[state]
        if self.gym_wrapping:
            action = self.env.action_space.from_gym(action)
        return action
