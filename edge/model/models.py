import numpy as np
import json
from pathlib import Path


class Model:
    """
    Base class for all Models. A Model is an object that can be queried and updated. The query should be done
    through the indexing syntax (model[i0,..., iN]), where the indexing should be thought of as if it were done on the
    StateActionSpace directly.
    """
    def __init__(self, env):
        """
        :param env: the environment
        """
        self.env = env

    def update(self):
        """ Abstract method
        Updates the model with new measurements. Subclasses may require more parameters.
        """
        raise NotImplementedError

    def _query(self, query):
        """ Abstract method
        Does all the processing to compute the query. A query comes in a form that is easy to handle for evaluation
        by the internal model: see _get_query_from_index to see the exact format. Input and output types should be
        defined by subclasses. Subclasses may require or permit additional arguments.
        :param query: the query
        :return: the evaluations of the model
        """
        raise NotImplementedError

    def _get_query_from_index(self, index):
        """ Abstract method
        Formats a StateActionSpace index into something understandable for the internal model representation.
        :param index: the index the model was called with. Should be the same format as a StateActionSpace index.
        :return: a format understandable by the internal model. Should be precised by subclasses
        """
        raise NotImplementedError

    def query(self, index, *args, **kwargs):
        """ Queries the model on the passed index, with optional additional argument.
        In general, this method should not be redefined: you probably want to redefine _get_query_from_index and/or
        _query instead.
        :param index: the index the model was called with. Should be the same format as a StateActionSpace index.
        :return: the evaluations of the model
        """
        query = self._get_query_from_index(index)
        return self._query(query, *args, **kwargs)

    def __getitem__(self, index):
        return self.query(index)


class DiscreteModel(Model):
    """
    Model on a Discrete StateActionSpace
    """
    def _get_query_from_index(self, index):
        """
        Transforms a StateActionSpace index into a np.ndarray index
        :param index: tuple: the index
        :return: a tuple that can index a np.ndarray. The tuple has d items, and each item is the list of indexes
            on the given dimension
        """
        stateactions = self.env.stateaction_space[index].reshape(
            (-1, self.env.stateaction_space.data_length)
        )
        if stateactions.ndim == 1:
            stateactions = np.atleast_2d(stateactions).tolist()
        index = np.array(list(map(
            self.env.stateaction_space.get_index_of,
            stateactions
        )))
        # `index` is now a list of n indexes, each of length d (index.shape = (n,d))
        # We need d lists of n points to index on a np.ndarray (or use the `np.take` method)
        # This is achieved by taking the transpose of index
        return tuple(index.T)


class ContinuousModel(Model):
    def _get_query_from_index(self, index):
        """
        Transforms a StateActionSpace index into a list of stateactions.
        :param index: tuple: the index
        :return: np.ndarray of stateactions
        """
        return self.env.stateaction_space[index].reshape(
            (-1, self.env.stateaction_space.data_length)
        )


class GPModel(ContinuousModel):
    GP_SAVE_NAME = 'gp.pth'  # name of file containing the GP when saving
    SAVE_NAME = 'model.json'  # name of file containing the Model's metadata when saving

    def __init__(self, env, gp):
        """
        Initializer
        :param env: the environment
        :param gp: the GP model, an instance of a subclass of edge.model.inference.GP. Should be initialized by
            a subclass
        """
        super(GPModel, self).__init__(env)
        self.gp = gp

    def _query(self, x):
        """
        Calls the GP model on the passed list of points
        :param x: a list of points on which to call the GP
        :return: the mean value of the GP at these points
        """
        return self.gp.predict(x).mean.numpy()

    def fit(self, train_x, train_y, epochs, **optimizer_kwargs):
        """
        Fits the GP's hyperparameters to the data. After training, the GP's dataset is reset to what it was before
        training.
        :param train_x: the input data to the GP
        :param train_y: the output data to the GP corresponding to the input data (targets)
        :param epochs: number of epochs
        :param optimizer_kwargs: parameters of the optimizer. Note: changing the optimizer itself can be done by setting
            self.gp.optimizer. Please use a PyTorch optimizer.
        """
        x_data = self.gp.train_x
        y_data = self.gp.train_y

        self.gp.set_data(train_x, train_y)
        self.gp.optimize_hyperparameters(epochs=epochs, **optimizer_kwargs)

        self.gp.set_data(x_data, y_data)

    def empty_data(self):
        """
        Empties the dataset of the GP.
        Note: this makes GPyTorch likely to fail.
        """
        # GPyTorch seems to have problems with empty datasets, so we keep a point very far away.
        # Note 1: with nonlocal kernels (CosineKernel for example), this may cause unexpected behaviour !
        # Note 2: the reason why GPyTorch fails with empty datasets seems to be different than the reason why
        #   using a 'far away' point also fails. There are two different failure cases here.
        pseudo_empty_x = np.ones(self.gp.input_shape) * (-1000)
        pseudo_empty_y = np.zeros(self.gp.output_shape)

        self.gp.set_data(pseudo_empty_x, pseudo_empty_y)

    def set_data(self, train_x, train_y):
        """
        Sets the data of the GP
        :param train_x:
        :param train_y:
        :return:
        """
        self.gp.set_data(train_x, train_y)

    @property
    def state_dict(self):
        """
        Dictionary of the configuration of the model. Useful for saving purposed
        :return: dict: the model's configuration
        """
        raise NotImplementedError

    def save(self, save_folder):
        """
        Saves the model in the given folder. The GP is saved in the file GPModel.GP_SAVE_NAME, and the model itself in
        GPModel.SAVE_NAME.
        :param save_folder: str or Path: the folder where to save
        """
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
        """
        Saves the data of the GP in npz format
        :param save_path: str or Path: the file where to save
        """
        save_path = str(save_path)
        np.savez(
            save_path,
            inputs=self.gp.train_x.numpy(),
            targets=self.gp.train_y.numpy()
        )

    def load_samples(self, load_path):
        """
        Loads data saved by the self.save_samples method and sets the GP dataset with it
        :param save_path: str or Path: the file where to save
        """
        data = np.load(load_path, allow_pickle=False)
        train_x = data['inputs']
        train_y = data['targets']
        self.set_data(train_x, train_y)

    @staticmethod
    def load(load_folder):
        """ Abstract method
        Loads the model and the GP saved by the GPModel.save method. Note that this method may fail if the save was
        made with an older version of the code.
        :param load_folder: the folder where the files are
        :return: GPModel: the model
        """
        # This method is abstract because it requires to dynamically intialize objects: knowing the exact class of
        # the GPModel is necessary to call its constructor
        raise NotImplementedError
