import numpy as np

from safe_rl.utils.load_utils import load_policy

from .. import ContinuousModel
from . import Policy
from edge.gym_wrappers import GymEnvironmentWrapper


class MultilayerPerceptron(ContinuousModel):
    def __init__(self, env, get_action):
        super(MultilayerPerceptron, self).__init__(env)
        self.get_action = get_action

    @staticmethod
    def load(load_folder, env):
        load_folder = str(load_folder)
        gym_env, get_action, sess = load_policy(load_folder)
        net = MultilayerPerceptron(env, get_action)
        return net, gym_env

    def update(self, *args, **kwargs):
        raise NotImplementedError('MultilayerPerceptron models are trained with the safety-starter-agents code, '
                                  'and edge does not support their training.')

    def _get_query_from_index(self, index):
        query = self.env.state_space[index].reshape(
            (-1, self.env.state_space.data_length)
        )
        return list(query)

    def _query(self, query):
        outputs = [self.get_action(query_item) for query_item in query]
        return np.array(outputs)


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
