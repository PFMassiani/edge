import logging
import gpytorch
import torch
from sklearn.neighbors import KDTree

from edge.utils import atleast_2d, dynamically_import, device, cuda
from .tensorwrap import tensorwrap, ensure_tensor
from edge.model.inference.kernels.value_structure_kernel import ValueStructureKernel, ValueStructureMean


def data_path_from_gp_path(gp_path):
    return gp_path[:-4] + '_data.pt'


class GP(gpytorch.models.ExactGP):
    """
    Base class for Gaussian Processes. Provides a wrapping around GPyTorch to encapsulate it from the rest of the code.
    """
    @tensorwrap('train_x', 'train_y')
    def __init__(self, train_x, train_y, mean_module, covar_module,
                 likelihood, dataset_type=None, dataset_params=None,
                 value_structure_discount_factor=None):
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
        :param value_structure_discount_factor: None or in (0,1). If float,
            wraps covar_module with a ValueStructureKernel with the given
            discount factor.
        """
        # The @tensorwrap decorator automatically transforms train_x and train_y into torch.Tensors.
        # Hence, we only deal with tensors inside the methods
        train_x = atleast_2d(train_x)
        train_y = train_y
        self.has_value_structure = value_structure_discount_factor is not None
        if dataset_type == 'timeforgetting':
            create_dataset = TimeForgettingDataset
        elif dataset_type == 'neighborerasing':
            create_dataset = NeighborErasingDataset
        else:
            dataset_params = {}
            create_dataset = Dataset
        dataset_params['has_is_terminal'] = self.has_value_structure
        self._dataset = create_dataset(train_x, train_y, **dataset_params)

        super(GP, self).__init__(train_x, train_y, likelihood)

        if self.has_value_structure:
            mean_module = ValueStructureMean(base_mean=mean_module)
            covar_module = ValueStructureKernel(
                base_kernel=covar_module,
                discount_factor=value_structure_discount_factor,
                dataset=self.dataset,
            )
        self.mean_module = mean_module
        self.covar_module = covar_module

        self.optimizer = torch.optim.Adam
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood

        self.to(device)
        self.likelihood.to(device)

    def initialize(self, **kwargs):
        if self.has_value_structure:
            change_info = [('covar_module.', ['covar_module', 'base_kernel']),
                           ('mean_module.', ['mean_module', 'base_mean'])]
            for current_start, new_start in change_info:
                kwargs_keys_to_change = [
                    key for key in kwargs.keys()
                    if key.startswith(current_start)
                ]
                for key in kwargs_keys_to_change:
                    key_parts = key.split('.')
                    key_parts = new_start + key_parts[1:]
                    new_key = '.'.join(key_parts)
                    kwargs[new_key] = kwargs.pop(key)
        super(GP, self).initialize(**kwargs)

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

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, new_dataset):
        self._dataset = new_dataset
        if self.has_value_structure:
            self.covar_module.dataset = new_dataset

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
        data_to_explain = self.train_y
        if self.has_value_structure:
            # We discard the last observation if the GP has a value structure
            data_to_explain = self.train_y[:-1]
        for n in range(epochs):
            optimizer.zero_grad()
            output = self(self.train_x)
            loss = -mll(output, data_to_explain)
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
        empty_x = torch.empty((0, self.train_x.shape[1]), device=device)
        empty_y = torch.empty(outputs_shape, device=device)
        self.train_x = empty_x
        self.train_y = empty_y
        if self.dataset.has_is_terminal:
            self.dataset.is_terminal = None
        self._set_gp_data_to_dataset()
        return self

    @tensorwrap('x', 'y')
    def set_data(self, x, y, **kwargs):
        """
        Sets the dataset of the GP
        :param x: np.ndarray: the new training input data. Subject to the same constraints as train_x in __init__
        :param y: np.ndarray: the new training output data. Subject to the same constraints as train_y in __init__
        """
        self.train_x = atleast_2d(x)
        self.train_y = y
        if self.dataset.has_is_terminal:
            self.dataset.is_terminal = kwargs.get('is_terminal')
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

    def save(self, save_path, save_data=False):
        """
        Saves the GP in PyTorch format, and optionally the Dataset object.
        PyTorch does NOT save samples or class structure. Such a model cannot
        be loaded by a simple "file.open" method.
        See the GP.load method for more information.
        :param save_path: str or Path: where to save the GP model
        :param save_data: bool: whether to save the Dataset as well
        """
        gp_save_path = str(save_path)
        if not save_path.endswith('.pth'):
            gp_save_path += '.pth'

        # In order to be able to load dynamically the model - that is, load it from the GP.load method - we need to know
        # the name of the true class of the GP
        save_dict = {
            'state_dict': self.state_dict(),
            'structure_dict': self.structure_dict,
            'classname': type(self).__name__
        }

        torch.save(save_dict, save_path)

        if save_data:
            data_save_path = data_path_from_gp_path(gp_save_path)
            self.dataset.save(data_save_path)

    # Careful: composing decorators with @staticmethod can be tricky. The @staticmethod decorator should be the last
    # one, because it does NOT return a method but an observer object
    @staticmethod
    @tensorwrap('train_x', 'train_y')
    def load(load_path, train_x, train_y, load_data=False):
        """
        Loads a model saved by the GP.save method, and sets its dataset with train_x, train_y. If `load_dataset` evaluates to true, it will then load and replace with a saved dataset.
        This method may fail if the GP was saved with an older version of the code.
        :param load_path: str or Path: the path to the file where the GP is saved
        :param train_x: np.ndarray: training input data. Should be 2D, and interpreted as a list of points.
        :param train_y: np.ndarray: training output data. Should be 1D, or of shape (train_x.shape[0], 1).
        :param load_dataset: optional str or Path to a file where the dataset is saved.
        :return: GP: an instance of the appropriate subclass of GP
        """
        load_path = str(load_path)
        save_dict = torch.load(load_path, map_location=device)
        classname = save_dict['classname']

        if load_data:
            data_path = data_path_from_gp_path(load_path)
            ds = Dataset.load(data_path)
            train_x = ds.train_x
            train_y = ds.train_y

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

        if load_data:
            model.dataset = ds

        return model


class Dataset:
    """
    Base class to handle datasets seamlessly in a GP. The properties `train_x` and `train_y` of a GP actually call the
    getters/setters of this class or a subclass of it, and the processing of the dataset is totally transparent from
    the GP class.
    Defining a custom way of handling GP data means inheriting from this class and redefining its getters and setters
    """
    def __init__(self, train_x, train_y, **kwargs):
        self.train_x = train_x
        self.train_y = train_y
        self.has_is_terminal = kwargs.get('has_is_terminal', False)
        if self.has_is_terminal:
            self.is_terminal = self._get_is_terminal(
                kwargs, self.train_y.shape[0], default=False
            )

    def _default_is_terminal(self, n, default=False):
        if default == False:
            default_constr = torch.zeros
        else:
            default_constr = torch.ones
        return default_constr(n, dtype=torch.bool, device=device)

    def _get_is_terminal(self, kwargs, n_default, default=False):
        return ensure_tensor(
            kwargs.get('is_terminal', self._default_is_terminal(
                n_default, default=default
            )),
            dtype=torch.bool,
        )

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

    @property
    def is_terminal(self):
        if not self.has_is_terminal:
            raise AttributeError
        else:
            return self._is_terminal

    @is_terminal.setter
    def is_terminal(self, new_is_terminal):
        self.has_is_terminal = True
        if new_is_terminal is None:
            new_is_terminal = self._default_is_terminal(self.train_y.shape[0])
        else:
            new_is_terminal = ensure_tensor(new_is_terminal)
        self._is_terminal = new_is_terminal

    def append(self, append_x, append_y, **kwargs):
        self.train_x = torch.cat((self.train_x, atleast_2d(append_x)), dim=0)
        self.train_y = torch.cat((self.train_y, append_y), dim=0)
        if self.has_is_terminal:
            self.is_terminal = torch.cat((
                self.is_terminal,
                self._get_is_terminal(kwargs, append_y.shape[0])
            ))

    def save(self, save_path):
        save_path = str(save_path)
        if not save_path.endswith('.pt'):
            save_path += '.pt'
        save_dict = {
            'train_x': self.train_x,
            'train_y': self.train_y,
        }
        if self.has_is_terminal:
            save_dict['is_terminal'] = self.is_terminal
        torch.save(save_dict, save_path)

    @staticmethod
    def load(load_path):
        load_path = str(load_path)
        save_dict = torch.load(load_path, map_location=device)
        ds = Dataset(save_dict.pop('train_x'), save_dict.pop('train_y'))
        # Dynamically set attributes so we allow loading attributes other than
        # train_x and train_y (like is_terminal)
        for aname, aval in save_dict.items():
            setattr(ds, aname, aval)
        return ds


class TimeForgettingDataset(Dataset):
    """
    This Dataset only keeps the last `keep` data points in the dataset, by simple assignment of the train_x and train_y
    attributes
    """
    def __init__(self, train_x, train_y, keep, **kwargs):
        """
        Initializer
        :param train_x: torch.Tensor, the training input data
        :param train_y: torch.Tensor, the training output data
        :param keep: int, how many points to keep in
        """
        self.keep = keep
        super(TimeForgettingDataset, self).__init__(train_x, train_y, **kwargs)
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

    @Dataset.is_terminal.setter
    def is_terminal(self, new_is_terminal):
        self.has_is_terminal = True
        if new_is_terminal is None:
            new_is_terminal = self._default_is_terminal(self.train_y.shape[0])
        else:
            new_is_terminal = ensure_tensor(new_is_terminal)
        self._is_terminal = new_is_terminal[-self.keep:]


class NeighborErasingDataset(Dataset):
    """
    When receiving a new point, this dataset removes the neighbors of that point that are close enough to it
    Note: this dataset does NOT ensure any condition when assigning a new set
    of points to train_x/train_y. The non-neighboring condition is only enforced
    when using the `append` method.
    """
    def __init__(self, train_x, train_y, radius, **kwargs):
        """
        Initializer
        :param train_x: torch.Tensor, the training input data
        :param train_y: torch.Tensor the training output data
        :param radius: float, the maximal distance at which old points will be removed when sampling a new point
        """
        if device == cuda:
            logging.warning('You are using a NeighborErasingDataset on the GPU.'
                            ' This Dataset uses scipy, which does not allow '
                            'parallelization: expect poor performance.')
        super(NeighborErasingDataset, self).__init__(train_x, train_y, **kwargs)
        self.radius = radius
        self.forgettable = torch.ones(self.train_x.shape[0], dtype=bool, device=device)
        self._kdtree = self._create_kdtree()

    def append(self, append_x, append_y, **kwargs):
        forgettable = kwargs.get('forgettable')
        if forgettable is None:
            forgettable = torch.ones(append_x.shape[0], dtype=bool,
                                     device=device)
        else:
            forgettable = torch.tensor(forgettable, dtype=bool, device=device)
        make_forget = kwargs.get('make_forget')
        if make_forget is None:
            make_forget = torch.ones(append_x.shape[0], dtype=bool,
                                     device=device)
        else:
            make_forget = torch.tensor(make_forget, dtype=bool, device=device)

        if make_forget.any():
            lists_indices_to_forget = self._kdtree.query_radius(
                append_x[make_forget].cpu().numpy(),
                r=self.radius
            )
            lists_indices_to_forget = [
                ensure_tensor(indices_to_forget, torch.long)
                for indices_to_forget in lists_indices_to_forget
            ]
            indices_to_forget = torch.cat(lists_indices_to_forget)
        else:
            indices_to_forget = ensure_tensor([], torch.long)

        to_forget_is_forgettable = self.forgettable[indices_to_forget]
        indices_to_forget = indices_to_forget[to_forget_is_forgettable]

        keeping_filter = torch.ones(self.train_x.shape[0], dtype=bool,
                                    device=device)
        keeping_filter[indices_to_forget] = False

        train_x_without_neighbors = self.train_x[keeping_filter]
        self.train_x = torch.cat((train_x_without_neighbors, append_x))

        train_y_without_neighbors = self._train_y[keeping_filter]
        self.train_y = torch.cat((train_y_without_neighbors, append_y))

        if self.has_is_terminal:
            append_is_terminal = self._get_is_terminal(
                kwargs, append_y.shape[0]
            )
            is_terminal_without_neighbors = self.is_terminal[keeping_filter]
            self.is_terminal = torch.cat(
                (is_terminal_without_neighbors, append_is_terminal)
            )

        forgettable_without_neighbors = self.forgettable[keeping_filter]
        self.forgettable = torch.cat((forgettable_without_neighbors,
                                      forgettable))

        self._kdtree = self._create_kdtree()

    def _create_kdtree(self):
        kdtree = KDTree(self.train_x.cpu().numpy(), leaf_size=40)  # Expensive
        return kdtree
