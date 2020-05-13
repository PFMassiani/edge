import numpy as np
from itertools import product


class Model:
    def __init__(self, space):
        self.space = space

    def update(self):
        raise NotImplementedError

    def query(self):
        raise NotImplementedError

    def _get_query_from_index(self, index):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.query(*args, **kwargs)

    def __getitem__(self, index):
        query = self._get_query_from_index(index)
        return self.query(query)


class DiscreteModel(Model):
    def _get_query_from_index(self, index):
        state, action = self.space.get_index_tuple(index)
        if isinstance(state, np.ndarray):
            state = self.space.state_space.get_index_of(state)
        if isinstance(action, np.ndarray):
            action = self.space.state_space.get_index_of(action)
        return tuple(np.hstack((state, action)))


class ContinuousModel(Model):
    def _get_query_from_index(self, index):
        state, action = self.space.get_index_tuple(index)
        if isinstance(state, np.ndarray):
            states = state
        else:
            states = self.space.state_space[state]
        if isinstance(action, np.ndarray):
            actions = action
        else:
            actions = self.space.action_space[action]
        states = np.atleast_2d(states)
        actions = np.atleast_2d(actions)
        query = np.zeros(
            (states.shape[1] * actions.shape[1], self.space.data_length),
            dtype=np.float
        )
        qind = 0
        for s, a in product(states, actions):
            query[qind] = self.space.get_stateaction(s, a)
            qind += 1
        return query


class GPModel(ContinuousModel):
    def __init__(self, space, gp):
        super(GPModel, self).__init__(self, space)
        self.gp = gp

    def query(self, x):
        return self.gp.predict(x).mean.numpy()
