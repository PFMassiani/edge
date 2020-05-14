import numpy as np
from itertools import product


class Model:
    def __init__(self, env):
        self.env = env

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
        state, action = self.env.stateaction_space.get_index_tuple(index)
        if isinstance(state, np.ndarray):
            state = self.env.stateaction_space.state_space.get_index_of(state)
        if isinstance(action, np.ndarray):
            action = self.env.stateaction_space.state_space.get_index_of(
                action)
        return tuple(np.hstack((state, action)))


class ContinuousModel(Model):
    def _get_query_from_index(self, index):
        state, action = self.env.stateaction_space.get_index_tuple(index)
        if isinstance(state, np.ndarray):
            states = state
        else:
            states = self.env.stateaction_space.state_space[state]
        if isinstance(action, np.ndarray):
            actions = action
        else:
            actions = self.env.stateaction_space.action_space[action]
        states = np.atleast_2d(states)
        actions = np.atleast_2d(actions)
        query = np.zeros(
            (
                states.shape[0] * actions.shape[0],
                self.env.stateaction_space.data_length
            ),
            dtype=np.float
        )
        qind = 0
        for s, a in product(states, actions):
            query[qind] = self.env.stateaction_space.get_stateaction(s, a)
            qind += 1
        return query


class GPModel(ContinuousModel):
    def __init__(self, env, gp):
        super(GPModel, self).__init__(env)
        self.gp = gp

    def query(self, x):
        return self.gp.predict(x).mean.numpy()
