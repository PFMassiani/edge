import gpytorch
import torch

from edge.utils import atleast_2d
from .tensorwrap import tensorwrap


class GPModel(gpytorch.models.ExactGP):
    @tensorwrap('train_x', 'train_y')
    def __init__(self, train_x, train_y, mean_module, covar_module,
                 likelihood):
        self.train_x = atleast_2d(train_x)
        self.train_y = train_y

        self.mean_module = mean_module
        self.covar_module = covar_module

        self.optimizer = torch.optim.Adam
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood

    @property
    def structure_dict(self):
        raise NotImplementedError

    @tensorwrap
    def __call__(self, *args, **kwargs):
        return super(GPModel, self).__call__(*args, **kwargs)

    @tensorwrap('x')
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def optimize_hyperparameters(self, epochs, **optimizer_kwargs):
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

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if gp_only:
                return self(x)
            else:
                return self.likelihood(self(x))

    def empty_data(self):
        outputs_shape = (0,) if len(self.train_y) == 1 else \
                        (0, self.train_y.shape[1])
        return self.set_data(
            inputs=torch.empty((0, self.train_x.shape[1])),
            outputs=torch.empty(outputs_shape)
        )

    @tensorwrap('x', 'y')
    def set_data(self, x, y):
        x = atleast_2d(x)

        self.train_x = x
        self.train_y = y
        self.set_train_data(
            inputs=self.train_x,
            outputs=self.train_y
        )
        return self

    @tensorwrap('x', 'y')
    def append_data(self, x, y):
        new_self = self.get_fantasy_model(x, y)
        new_self.train_x = atleast_2d(x)
        new_self.train_y = y
        return new_self

    def save(self, save_path):
        if not save_path.endswith('.pth'):
            save_path += '.pth'

        save_dict = {
            'state_dict': self.state_dict,
            'structure_dict': self.structure_dict,
            'classname': type(self).__name__
        }

        torch.save(save_dict, save_path)

    @staticmethod
    @tensorwrap('train_x', 'train_y')
    def load(load_path, train_x, train_y):
        save_dict = torch.load(load_path)
        classname = save_dict['classname']

        constructor = globals().get(classname)
        if constructor is None:
            raise NameError(f'Name {classname} is not defined')
        construction_parameters = save_dict['structure_dict']
        model = constructor(
            train_x=train_x,
            train_y=train_y,
            **construction_parameters
        )
        model.load_state_dict(save_dict['state_dict'])
        return model
