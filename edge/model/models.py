import numpy as np
from itertools import product


class Model:
    def __init__(self, env):
        self.env = env

    def update(self):
        raise NotImplementedError

    def _query(self):
        raise NotImplementedError

    def _get_query_from_index(self, index):
        raise NotImplementedError

    def query(self, index, *args, **kwargs):
        query = self._get_query_from_index(index)
        return self._query(query, *args, **kwargs)

    def __getitem__(self, index):
        return self.query(query)


class DiscreteModel(Model):
    def _get_query_from_index(self, index):
        stateactions = self.env.stateaction_space[index]
        if stateactions.ndim == 1:
            stateactions = np.atleast_2d(stateactions)
        index = np.array(list(map(
            self.env.stateaction_space.get_index_of,
            stateactions
        )))
        # Index is a nxd list of indexes we will use to index on a np.ndarray
        # We need d lists of n points for the function take or the [] operator
        # This is achieved by taking the transpose of index
        return index.T.tolist()


class ContinuousModel(Model):
    def _get_query_from_index(self, index):
        return self.env.stateaction_space[index].reshape(
            (-1, self.env.stateaction_space.data_length)
        )


class GPModel(ContinuousModel):
    def __init__(self, env, gp):
        super(GPModel, self).__init__(env)
        self.gp = gp

    def _query(self, x):
        return self.gp.predict(x).mean.numpy()
