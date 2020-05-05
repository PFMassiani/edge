from itertools import product, starmap
import numpy as np

from edge.utils import ensure_list


class Space:
    def __init__(self):
        pass

    def contains(self, x):
        raise NotImplementedError

    def __contains__(self, x):
        return self.contains(x)

    def sample(self):
        raise NotImplementedError


class DiscreteSpace(Space):
    def __init__(self, index_dim):
        super(DiscreteSpace, self).__init__()
        self.index_dim = index_dim

    def __getitem__(self, index):
        raise NotImplementedError

    def indexof(self, x):
        raise NotImplementedError

    def __iter__(self):
        return self.get_index_iterator()

    def get_index_iterator(self):
        raise NotImplementedError

    def sample_idx(self):
        raise NotImplementedError

    def sample(self):
        k = self.sample_idx()
        return self[k]


class DiscreteProductSpace(DiscreteSpace):
    def __init__(self, *sets):
        self.sets = sets
        self._n_sets = len(self.sets)
        index_dim = 0
        for s in self.sets:
            index_dim += s.index_dim

        super(DiscreteProductSpace, self).__init__(index_dim)

        self._index_masks = [None] * self._n_sets
        current_index = 0
        for ns in range(self._n_sets):
            end_index = current_index + self.sets[ns].index_dim
            self._index_masks[ns] = slice(current_index, end_index)
            current_index = end_index

    def contains(self, x):
        if len(x) != self.index_dim:
            raise ValueError(f"Size mismatch: expected size {self.index_dim}"
                             f", got {len(x)}")
        isin = True
        for ns in range(self._n_sets):
            s = self.sets[ns]
            mask = self._index_masks[ns]
            isin = isin and (x[mask] in s)

        return isin

    def sample_idx(self):
        def sample_set(s):
            index = s.sample_idx()
            index = np.array(ensure_list(index), dtype=np.int).reshape(-1)
            return index
        return np.concatenate(tuple(map(sample_set, self.sets)))

    def __getitem__(self, index):
        if len(index) != self.index_dim:
            raise ValueError(f'Size mismatch: expected size {len(self.sets)}'
                             f', got {len(index)}')
        x = []
        for ns in range(self._n_sets):
            mask = self._index_masks[ns]
            index_in_set = index[mask]
            s = self.sets[ns]
            x = np.concatenate((x, s[index_in_set]))
        return x

    def indexof(self, x):
        if len(x) != self.index_dim:
            raise ValueError(f'Size mismatch: expected size {len(self.sets)}'
                             f', got {len(x)}')
        index = np.zeros(self.index_dim, dtype=np.int)
        for ns in range(self._n_sets):
            s = self.sets[ns]
            mask = self._index_masks[ns]
            index[mask] = s.indexof(x[mask])
        return index

    def get_index_iterator(self):
        def flatten(*args):
            x = []
            for y in args:
                z = ensure_list(y)
                x += z
            return np.array(x, dtype=np.int)
        return starmap(
            flatten,
            product(*self.sets)
        )


class DiscreteStateActionSpace(DiscreteProductSpace):
    def __init__(self, state_space, action_space):
        super(DiscreteStateActionSpace, self).__init__(
            state_space,
            action_space
        )
        self.state_space = state_space
        self.action_space = action_space


class DiscreteSubspace(DiscreteSpace):
    def __init__(self, space, idx_in_subspace):
        super(DiscreteSubspace, self).__init__()
        # TODO
