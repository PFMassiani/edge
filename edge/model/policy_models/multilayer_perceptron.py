import numpy as np

from safe_rl.utils.load_utils import load_policy

from .. import ContinuousModel


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