import pickle as pkl
import numpy as np

from .. import GroundTruth
from edge.space import Segment, ProductSpace, StateActionSpace


class SafetyTruth(GroundTruth):
    def __init__(self, env):
        super(SafetyTruth, self).__init__()
        self.env = env

        # These attributes are initialized either by compute or from_vibly_file
        self.stateaction_space = None
        self.viable_set = None
        self.measure = None

    def get_training_examples(self):
        examples = [(stateaction, self.measure[index])
                    for index, stateaction in iter(self.stateaction_space)]
        train_x, train_y = zip(*examples)

        return np.atleast_2d(train_x), np.atleast_2d(train_y)

    def from_vibly_file(self, vibly_file_path):
        with open(vibly_file_path, 'rb') as f:
            data = pkl.load(f)
        # dict_keys(['grids', 'Q_map', 'Q_F', 'Q_V', 'Q_M', 'S_M', 'p', 'x0'])
        states = data['grids']
        actions = data['actions']

        if len(states) != self.env.state_space.index_dim:
            raise IndexError('Size mismatch : expected state space with '
                             f'{self.env.state_space.index_dim} dimensions, '
                             f'got {len(states)}')
        if len(actions) != self.env.action_space.index_dim:
            raise IndexError('Size mismatch : expected action space with '
                             f'{self.env.action_space.index_dim} dimensions, '
                             f'got {len(actions)}')

        def get_product_space(np_components):
            segments = [Segment(comp[0], comp[-1], comp.shape[0])
                        for comp in np_components]
            product_space = ProductSpace(*segments)
            return product_space

        state_space = get_product_space(states)
        action_space = get_product_space(actions)
        self.stateaction_space = StateActionSpace(state_space, action_space)

        self.viable_set = data['Q_V']
        self.measure = data['Q_M']
