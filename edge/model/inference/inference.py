import gpytorch
import torch
from sklearn.neighbors import KDTree

from edge.utils import atleast_2d, dynamically_import
from .tensorwrap import tensorwrap


class GP(gpytorch.models.ExactGP):
    """
    Base class for Gaussian Processes. Provides a wrapping around GPyTorch to encapsulate it from the rest of the code.
    """
    @tensorwrap('train_x', 'train_y')
    def __init__(self, train_x, train_y, mean_module, covar_module,
                 likelihood, dataset_type=None, dataset_params=None):
        """
        Initializer
        :param train_x: np.ndarray: training input data. Should be 2D, and interpreted as a list of points.
        :param train_y: np.ndarray: training output data. Should be 1D, or of shape (train_x.shape[0], 1).
        :param mean_module: the mean of the GP. See GPyTorch tutorial
        :param covar_module: the covariance of the GP. See GPyTorch tutorial
        :param likelihood: the likelihood of the GP. See GPyTorch tutorial
        :param dataset_type: Possible values:
            * 'timeforgetting': use a TimeForgettingDataset
            * 'neighborerasing': use a NeighborErasingDataset
            * anything else: use a standard Dataset
        :param dataset_params: dictionary or None. The entries are passed as keyword arguments to the constructor of
            the chosen dataset.
        """
        # The @tensorwrap decorator automatically transforms train_x and train_y into torch.Tensors.
        # Hence, we only deal with tensors inside the methods
        train_x = atleast_2d(train_x)
        train_y = train_y
        if dataset_type == 'timeforgetting':
            create_dataset = TimeForgettingDataset
        elif dataset_type == 'neighborerasing':
            create_dataset = NeighborErasingDataset
        else:
            dataset_params = {}
            create_dataset = Dataset
        self.dataset = create_dataset(train_x, train_y, **dataset_params)

        super(GP, self).__init__(train_x, train_y, likelihood)

        self.mean_module = mean_module
        self.covar_module = covar_module

        self.optimizer = torch.optim.Adam
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood

    @property
    def train_x(self):
        return self.dataset.train_x

    @train_x.setter
    def train_x(self, new_train_x):
        self.dataset.train_x = new_train_x

    @property
    def train_y(self):
        return self.dataset.train_y

    @train_y.setter
    def train_y(self, new_train_y):
        self.dataset.train_y = new_train_y

    @property
    def structure_dict(self):
        """ Abstract property
        Should provide the dictionary of all the parameters necessary to initialize a GP. In practice, this dictionary
        is passed with the `**` operator to the constructor of the GP.
        This is required in order to load/save the GP.
        :return: the dictionary of parameters defining the structure of the GP
        """
        raise NotImplementedError

    @property
    def input_shape(self):
        shape_x = tuple(self.train_x.shape)
        return shape_x[1:]

    @property
    def output_shape(self):
        shape_y = tuple(self.train_y.shape)
        if len(shape_y) == 1:
            return (1,)
        else:
            return shape_y[1:]

    # This is a simple redefinition of super().__call__().
    # The goal here is to add the @tensorwrap decorator, so the parameters used to call the GP are automatically
    # transformed into tensors, because super().__call__ does not work with np.ndarrays
    @tensorwrap()
    def __call__(self, *args, **kwargs):
        return super(GP, self).__call__(*args, **kwargs)

    @tensorwrap('x')
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def optimize_hyperparameters(self, epochs, **optimizer_kwargs):
        """
        Optimizes the hyperparameters of the GP on its current dataset. This function can be run several times with
        different parameterizations for the optimizer, enabling refining of parameters such as the learning rate.
        If you want a "smart" adaptation of the parameters, you should redefine this method.
        :param epochs: the number of epochs
        :param optimizer_kwargs: the parameters passed to the optimizer. See GPyTorch documentation for more information
        """
        self.train()
        self.likelihood.train()

        optimizer = self.optimizer(
            [{'params': self.parameters()}],
            **optimizer_kwargs
        )
        mll = self.mll(self.likelihood, self)

        for n in range(epochs):
            optimizer.zero_grad()
            output = self(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()

    @tensorwrap('x')
    def predict(self, x, gp_only=False):
        self.eval()
        self.likelihood.eval()

        # This `with` clause is taken from the GPyTorch tutorials. I don't know whether they truly improve performance
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if gp_only:
                return self(x)
            else:
                return self.likelihood(self(x))

    def _set_gp_data_to_dataset(self):
        self.set_train_data(
            inputs=self.train_x,
            targets=self.train_y,
            strict=False
            # If True, the new dataset should have the same shape as the old one
        )

    def empty_data(self):
        """
        Empties the dataset of the GP
        Note: this makes GPyTorch likely to fail.
        """
        outputs_shape = (0,) if len(self.train_y.shape) == 1 else \
                        (0, self.train_y.shape[1])
        empty_x = torch.empty((0, self.train_x.shape[1]))
        empty_y = torch.empty(outputs_shape)
        self.train_x = empty_x
        self.train_y = empty_y
        self._set_gp_data_to_dataset()
        return self

    @tensorwrap('x', 'y')
    def set_data(self, x, y):
        """
        Sets the dataset of the GP
        :param x: np.ndarray: the new training input data. Subject to the same constraints as train_x in __init__
        :param y: np.ndarray: the new training output data. Subject to the same constraints as train_y in __init__
        """
        self.train_x = atleast_2d(x)
        self.train_y = y
        self._set_gp_data_to_dataset()
        return self

    @tensorwrap('x', 'y')
    def append_data(self, x, y, **kwargs):
        """
        Appends data to the GP dataset
        :param x: np.ndarray: the additional training input data. Should be 2D, with the same number of columns than
            self.train_x
        :param y: np.ndarray: the addition training output data. Should be of shape (x.shape[0],) or (x.shape[0], 1)
        """
        # GPyTorch provides an additional, more efficient way of adding data with the ExactGP.get_fantasy_model method,
        # but it seems to require that the model is called at least once before it can be used
        self.dataset.append(atleast_2d(x), y, **kwargs)
        self._set_gp_data_to_dataset()
        return self

    def save(self, save_path):
        """
        Saves the GP in PyTorch format, and optionally the Dataset object.
        PyTorch does NOT save samples or class structure. Such a model cannot
        be loaded by a simple "file.open" method.
        See the GP.load method for more information.
        :param save_path: str or Path: where to save the GP model
        """
        save_path = str(save_path)
        if not save_path.endswith('.pth'):
            save_path += '.pth'

        # In order to be able to load dynamically the model - that is, load it from the GP.load method - we need to know
        # the name of the true class of the GP
        save_dict = {
            'state_dict': self.state_dict(),
            'structure_dict': self.structure_dict,
            'classname': type(self).__name__
        }

        torch.save(save_dict, save_path)

    def save_dataset(self, save_path):
        """
        Saves a dataset, using the method implemented in the dataset class.
        :param save_data: str or Path: where to save the Dataset
        """
        save_path = str(save_path)
        if not save_path.endswith('.pth'):
            save_path += '.pth'
        self.dataset.save(save_path)

    # Careful: composing decorators with @staticmethod can be tricky. The @staticmethod decorator should be the last
    # one, because it does NOT return a method but an observer object
    @staticmethod
    @tensorwrap('train_x', 'train_y')
    def load(load_path, train_x, train_y):
        """
        Loads a model saved by the GP.save method, and sets its dataset with train_x, train_y.
        This method may fail if the GP was saved with an older version of the code.
        :param load_path: str or Path: the path to the file where the GP is saved
        :param train_x: np.ndarray: training input data. Should be 2D, and interpreted as a list of points.
        :param train_y: np.ndarray: training output data. Should be 1D, or of shape (train_x.shape[0], 1).
        :return: GP: an instance of the appropriate subclass of GP
        """
        load_path = str(load_path)
        save_dict = torch.load(load_path)
        classname = save_dict['classname']

        # We know the name of the true class of the GP, so we can dynamically import it. This is ugly and not robust,
        # but it avoids having to redefine the load method in every subclass
        constructor = dynamically_import('edge.model.inference.' + classname)
        if constructor is None:
            raise NameError(f'Name {classname} not found')
        construction_parameters = save_dict['structure_dict']
        model = constructor(
            train_x=train_x,
            train_y=train_y,
            **construction_parameters
        )
        model.load_state_dict(save_dict['state_dict'])

        return model

    def load_dataset(self, load_path):
        """
        Loads and sets `train_x` and `train_y`.
        :param load_path: str or Path: the path to the data file
        """
        load_path = str(load_path)
        self.dataset.load(load_path)
        self._set_gp_data_to_dataset()


class Dataset:
    """
    Base class to handle datasets seamlessly in a GP. The properties `train_x` and `train_y` of a GP actually call the
    getters/setters of this class or a subclass of it, and the processing of the dataset is totally transparent from
    the GP class.
    Defining a custom way of handling GP data means inheriting from this class and redefining its getters and setters
    """
    def __init__(self, train_x, train_y):
        self._train_x = train_x
        self._train_y = train_y

    @property
    def train_x(self):
        return self._train_x

    @train_x.setter
    def train_x(self, new_train_x):
        self._train_x = new_train_x

    @property
    def train_y(self):
        return self._train_y

    @train_y.setter
    def train_y(self, new_train_y):
        self._train_y = new_train_y

    def append(self, append_x, append_y, **kwargs):
        self.train_x = torch.cat((self.train_x, atleast_2d(append_x)), dim=0)
        self.train_y = torch.cat((self.train_y, append_y), dim=0)

    def save(self, save_path):
        save_path = str(save_path)
        if not save_path.endswith('.pth'):
            save_path += '.pth'

        torch.save({'train_x': self.train_x,
                    'train_y': self.train_y},
                   save_path)

    def load(self, load_path):
        load_path = str(load_path)
        save_dict = torch.load(load_path)
        self.train_x = save_dict['train_x']
        self.train_y = save_dict['train_y']


class TimeForgettingDataset(Dataset):
    """
    This Dataset only keeps the last `keep` data points in the dataset, by simple assignment of the train_x and train_y
    attributes
    """
    def __init__(self, train_x, train_y, keep):
        """
        Initializer
        :param train_x: torch.Tensor, the training input data
        :param train_y: torch.Tensor, the training output data
        :param keep: int, how many points to keep in
        """
        super(TimeForgettingDataset, self).__init__(train_x, train_y)
        self.keep = keep
        self._train_x = train_x[-self.keep:]
        self._train_y = train_y[-self.keep:]

    # These redefine the setters of train_x and train_y
    # Hence, we do NOT need to redefine TimeForgettingDataset.append, because
    # the setters already make sure that there is at most self.keep points
    @Dataset.train_x.setter
    def train_x(self, new_train_x):
        self._train_x = new_train_x[-self.keep:]

    @Dataset.train_y.setter
    def train_y(self, new_train_y):
        self._train_y = new_train_y[-self.keep:]


class NeighborErasingDataset(Dataset):
    """
    When receiving a new point, this dataset removes the neighbors of that point that are close enough to it
    Note: this dataset does NOT ensure any condition when assigning a new set
    of points to train_x/train_y. The non-neighboring condition is only enforced
    when using the `append` method.
    """
    def __init__(self, train_x, train_y, radius):
        """
        Initializer
        :param train_x: torch.Tensor, the training input data
        :param train_y: torch.Tensor the training output data
        :param radius: float, the maximal distance at which old points will be removed when sampling a new point
        """
        super(NeighborErasingDataset, self).__init__(train_x, train_y)
        self.radius = radius
        self.forgettable = torch.ones(self.train_x.shape[0], dtype=bool)
        self._kdtree = self._create_kdtree()

    def append(self, append_x, append_y, **kwargs):
        forgettable = kwargs.get('forgettable')
        if forgettable is None:
            forgettable = torch.ones(append_x.shape[0], dtype=bool)
        else:
            forgettable = torch.tensor(forgettable, dtype=bool)
        make_forget = kwargs.get('make_forget')
        if make_forget is None:
            make_forget = torch.ones(append_x.shape[0], dtype=bool)
        else:
            make_forget = torch.tensor(make_forget, dtype=bool)

        if make_forget.any():
            lists_indices_to_forget = self._kdtree.query_radius(
                append_x[make_forget].numpy(),
                r=self.radius
            )
            lists_indices_to_forget = list(map(
                torch.tensor,
                lists_indices_to_forget
            ))
            indices_to_forget = torch.cat(lists_indices_to_forget)
        else:
            indices_to_forget = torch.tensor([], dtype=int)

        to_forget_is_forgettable = self.forgettable[indices_to_forget]
        indices_to_forget = indices_to_forget[to_forget_is_forgettable]

        keeping_filter = torch.ones(self.train_x.shape[0], dtype=bool)
        keeping_filter[indices_to_forget] = False

        train_x_without_neighbors = self.train_x[keeping_filter]
        self.train_x = torch.cat((train_x_without_neighbors, append_x))

        train_y_without_neighbors = self._train_y[keeping_filter]
        self.train_y = torch.cat((train_y_without_neighbors, append_y))

        forgettable_without_neighbors = self.forgettable[keeping_filter]
        self.forgettable = torch.cat((forgettable_without_neighbors,
                                      forgettable))

        self._kdtree = self._create_kdtree()

    def _create_kdtree(self):
        kdtree = KDTree(self.train_x.numpy(), leaf_size=40)  # This is expensive
        return kdtree