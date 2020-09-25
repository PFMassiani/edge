import numpy as np

from . import Agent
from edge.model.policy_models import MLPPolicy
from edge.model.policy_models import MultilayerPerceptron


class PolicyLearner(Agent):
    def __init__(self, env, policy_load_folder):
        policy_model, saved_env = MultilayerPerceptron.load(policy_load_folder,
                                                            env)
        self.policy = MLPPolicy(env, policy_model)
        super(PolicyLearner, self).__init__(env, policy_model)

    def get_next_action(self):
        return self.env.action_space.closest_in(self.policy.get_action(self.state))

    def update_models(self, state, action, new_state, reward, failed, done):
        pass