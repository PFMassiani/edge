import pickle as pkl
import numpy as np

from .. import GroundTruth
from edge.space import Segment, ProductSpace, StateActionSpace
from edge.utils import get_parameters_lookup_dictionary


class SafetyTruth(GroundTruth):
    def __init__(self, env):
        super(SafetyTruth, self).__init__()
        self.env = env

        # These attributes are initialized either by compute or from_vibly_file
        self.stateaction_space = None
        self.viable_set = None
        self.measure = None

    def get_training_examples(self, n_examples=2000, from_viable=True,
                              from_failure=False, viable_proportion=0.6):
        def sample_when_true(n, condition):
            if n > 0:
                idx_satisfying_condition = np.argwhere(condition)
            else:
                idx_satisfying_condition = np.empty(
                    shape=(0, len(condition.shape)),
                    dtype=int
                )

            n = min(n, len(idx_satisfying_condition))
            sample = np.random.choice(
                idx_satisfying_condition.shape[0], n, replace=False
            )
            sample_idx_list = idx_satisfying_condition[sample]
            sample_idx_tuple = tuple(zip(*sample_idx_list))
            if len(sample_idx_tuple) == 0:
                sample_idx_tuple = ((), ())

            sample_x = np.array([
                self.stateaction_space[tuple(idx)]
                for idx in sample_idx_list
            ]).reshape((-1, self.stateaction_space.index_dim))
            sample_y = self.measure[sample_idx_tuple].squeeze()
            return sample_x, sample_y

        n_viable = 0 if not from_viable\
            else n_examples if not from_failure\
            else int(viable_proportion * n_examples)
        n_failure = n_examples - n_viable

        viable_x, viable_y = sample_when_true(
            n_viable, self.viable_set.astype(bool)
        )
        failure_x, failure_y = sample_when_true(
            n_failure, self.failure_set.astype(bool)
        )

        train_x = np.vstack((viable_x, failure_x))
        train_y = np.hstack((viable_y, failure_y))

        return train_x, train_y

    def from_vibly_file(self, vibly_file_path):
        with open(vibly_file_path, 'rb') as f:
            data = pkl.load(f)
        # dict_keys(['grids', 'Q_map', 'Q_F', 'Q_V', 'Q_M', 'S_M', 'p', 'x0'])
        states = data['grids']['states']
        actions = data['grids']['actions']

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

        self.state_measure = data['S_M']
        self.measure = data['Q_M']
        self.viable_set = data['Q_V']
        self.failure_set = data['Q_F']
        self.unviable_set = ~self.failure_set
        self.unviable_set[self.viable_set] = False

        self.viable_set = self.viable_set.astype(float)
        self.failure_set = self.failure_set.astype(float)
        self.unviable_set = self.unviable_set.astype(float)

        lookup_dictionary = get_parameters_lookup_dictionary(self.env)

        for vibly_pname, vibly_value in data['p'].items():
            pname = lookup_dictionary[vibly_pname]
            if pname is not None:
                value = self.env.dynamics.parameters[pname]
                if value != vibly_value:
                    raise ValueError(
                        f'Value mismatch: value loaded for {pname} from '
                        f'{vibly_pname} does not match. Expected {value}, '
                        f'got {vibly_value}'
                    )
