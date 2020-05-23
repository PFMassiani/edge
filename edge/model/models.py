import numpy as np
import json
from pathlib import Path


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
        return self.query(index)


class DiscreteModel(Model):
    def _get_query_from_index(self, index):
        stateactions = self.env.stateaction_space[index].reshape(
            (-1, self.env.stateaction_space.data_length)
        )
        if stateactions.ndim == 1:
            stateactions = np.atleast_2d(stateactions).tolist()
        index = np.array(list(map(
            self.env.stateaction_space.get_index_of,
            stateactions
        )))
        # Index is a nxd list of indexes we will use to index on a np.ndarray
        # We need d lists of n points for the function take or the [] operator
        # This is achieved by taking the transpose of index
        return tuple(index.T)


class ContinuousModel(Model):
    def _get_query_from_index(self, index):
        return self.env.stateaction_space[index].reshape(
            (-1, self.env.stateaction_space.data_length)
        )


class GPModel(ContinuousModel):
    GP_SAVE_NAME = 'gp.pth'
    SAVE_NAME = 'model.json'

    def __init__(self, env, gp):
        super(GPModel, self).__init__(env)
        self.gp = gp

    def _query(self, x):
        return self.gp.predict(x).mean.numpy()

    def fit(self, train_x, train_y, epochs, **optimizer_kwargs):
        x_data = self.gp.train_x
        y_data = self.gp.train_y

        self.gp.set_data(train_x, train_y)
        self.gp.optimize_hyperparameters(epochs=epochs, **optimizer_kwargs)

        self.gp.set_data(x_data, y_data)

    def empty_data(self):
        pseudo_empty_x = np.ones(self.gp.input_shape) * (-1000)
        pseudo_empty_y = np.zeros(self.gp.output_shape)

        self.gp.set_data(pseudo_empty_x, pseudo_empty_y)

    def set_data(self, train_x, train_y):
        self.gp.set_data(train_x, train_y)

    @property
    def state_dict(self):
        raise NotImplementedError

    def save(self, save_folder):
        save_path = Path(save_folder)
        gp_save_path = str(save_path / GPModel.GP_SAVE_NAME)
        model_save_path = str(save_path / GPModel.SAVE_NAME)

        self.gp.save(gp_save_path)
        try:
            state_dict = self.state_dict
            with open(model_save_path, 'w') as f:
                json.dump(state_dict, f, indent=4)
        except NotImplementedError:
            pass

    def save_samples(self, save_path):
        np.savez(
            save_path,
            inputs=self.gp.train_x.numpy(),
            targets=self.gp.train_y.numpy()
        )

    def load_samples(self, load_path):
        data = np.load(load_path, allow_pickle=False)
        train_x = data['inputs']
        train_y = data['targets']
        self.set_data(train_x, train_y)

    @staticmethod
    def load(load_folder):
        raise NotImplementedError
