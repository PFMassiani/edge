import unittest
import torch
import numpy as np
import tempfile
import os
import warnings
import gpytorch
import matplotlib.pyplot as plt

from edge.model.inference import tensorwrap
from edge.model.inference.tensorwrap import ensure_tensor
from edge.model.inference import MaternGP, GP
from edge.model.inference.inference import NeighborErasingDataset
from edge.utils import constraint_from_tuple


class TestTensorwrap(unittest.TestCase):
    def check_tensor(self, *args, **kwargs):
        for a in args:
            self.assertTrue(torch.is_tensor(a))
        for k, a in kwargs.items():
            self.assertTrue(torch.is_tensor(a))

    def test_tensorwrap_for_method(self):
        @tensorwrap()
        def m1(x, y):
            self.check_tensor(x, y)

        @tensorwrap(1)
        def m2(x, y):
            self.check_tensor(x)
            self.assertFalse(torch.is_tensor(y))

        @tensorwrap('y')
        def m3(x, y):
            self.check_tensor(y)
            self.assertFalse(torch.is_tensor(x))

        @tensorwrap()
        def m4(*args, **kwargs):
            self.check_tensor(*args, **kwargs)

        t = np.array([0])
        m1(t, y=t)
        m2(t, t)
        m3(t, t)
        m4(t, key=t)

    def test_tensorwrap_for_class(me):
        class Foo:
            @tensorwrap()
            def m1(self, x, y):
                me.assertFalse(torch.is_tensor(self))
                me.check_tensor(x, y)

            @tensorwrap(1)
            def m2(self, x, y):
                me.check_tensor(x)
                me.assertFalse(torch.is_tensor(self))
                me.assertFalse(torch.is_tensor(y))

            @tensorwrap('y')
            def m3(self, x, y):
                me.check_tensor(y)
                me.assertFalse(torch.is_tensor(self))
                me.assertFalse(torch.is_tensor(x))

            @staticmethod
            @tensorwrap(1)
            def m4(x, y):
                me.check_tensor(x)
                me.assertFalse(torch.is_tensor(y))

            @tensorwrap()
            def m5(self, *args, **kwargs):
                me.assertFalse(torch.is_tensor(self))
                me.check_tensor(*args, **kwargs)

        bar = Foo()
        t = np.array([0])

        bar.m1(t, y=t)
        bar.m2(t, t)
        bar.m3(t, t)
        bar.m4(t, t)
        Foo.m4(t, t)
        bar.m5(t, key=t)


class TestGP(unittest.TestCase):
    def _matern_52(self, x, outputscale, lengthscale):
        return outputscale ** 2 * (
            1 + np.sqrt(5) * x /lengthscale+ 5 * (x ** 2)/(3*lengthscale**2)
        ) * np.exp(
            - np.sqrt(5) * x / lengthscale
        )

    def _heaviside(self, x):
        y = np.zeros_like(x)
        y[x >= 0] = 1
        return y

    def test_initialization(self):
        x = np.arange(18).reshape((6, 3))
        y = np.arange(6)
        gp = MaternGP(x, y)
        self.assertTrue(torch.is_tensor(gp.train_x))
        self.assertTrue(torch.is_tensor(gp.train_y))

    def test_load_save(self):
        x = np.linspace(0, 1, 11)
        y = np.sin(2 * np.pi * x) + np.random.randn(len(x)) * 0.2

        model = MaternGP(
            x,
            y,
            noise_prior=(1, 0.1),
            noise_constraint=0.5,
            outputscale_constraint=(2.7, np.pi)
        )
        save_file = tempfile.NamedTemporaryFile(suffix='.pth').name
        model.save(save_file)
        self.assertTrue(os.path.isfile(save_file))

        loaded = MaternGP.load(save_file, x, y)

        self.assertEqual(model.covar_module.outputscale,
                         loaded.covar_module.outputscale)

        save_data = tempfile.NamedTemporaryFile(suffix='.pth').name
        model.save(save_file, save_data=save_data)
        self.assertTrue(os.path.isfile(save_file))
        self.assertTrue(os.path.isfile(save_data))
        x2 = np.linspace(2, 3, 11)
        loaded = MaternGP.load(save_file, x2, y, save_data)
        self.assertTrue(torch.all(torch.eq(model.train_x, loaded.train_x)))

    def test_hyper_optimization_0(self):
        warnings.simplefilter('ignore', gpytorch.utils.warnings.GPInputWarning)

        tol = 1e-3
        x = np.linspace(0, 1, 101).reshape((-1, 1))
        y = np.exp(-x**2).reshape(-1)

        gp = MaternGP(x, y, noise_constraint=(0, 1e-4))
        gp.optimize_hyperparameters(epochs=20)
        gp.optimize_hyperparameters(epochs=20, lr=0.01)

        predictions = gp.predict(x)
        y_pred = predictions.mean.numpy()

        passed = np.all(np.abs(y - y_pred) < tol)

        if passed:
            self.assertTrue(True)
        else:
            x = x.reshape(-1)
            f, ax = plt.subplots(1, 1, figsize=(4, 3))
            lower, upper = predictions.confidence_region()
            ax.plot(x, y, 'k*')
            ax.plot(x, y_pred, 'b')
            ax.fill_between(x, lower.numpy(), upper.numpy(), alpha=0.5)
            ax.legend(['Observed Data', 'Mean' , 'Confidence','Noise'])
            ax.grid(True)
            plt.show()
            self.assertTrue(False)

    def test_hyper_optimization_1(self):
        warnings.simplefilter('ignore', gpytorch.utils.warnings.GPInputWarning)

        tol = 5e-3
        lengthscale = 1.5
        outputscale = 2
        x = np.linspace(-1, 1, 501).reshape((-1, 1))
        # Eventhough the data is generated by a Matern function, the lengthscale and outputscale learned by the Matern
        # kernel GP do NOT need to coincide with the ones used to generate the data: the learned ones correspond to the
        # influence of nearby points, not to the global structure of the data
        y = self._matern_52(x, lengthscale=lengthscale, outputscale=outputscale).reshape(-1)
        x_train = x[::2]
        y_train = y[::2]
        x_test = x[1::2]
        y_test = y[1::2]

        gp = MaternGP(x_train, y_train, nu=5/2, noise_constraint=(0,1e-5), hyperparameters_initialization={
            'covar_module.base_kernel.lengthscale': 1,
            'covar_module.outputscale': 1
        })
        gp.optimizer = torch.optim.Adam
        gp.optimize_hyperparameters(epochs=50, lr=0.01)
        gp.optimize_hyperparameters(epochs=50, lr=0.001)
        gp.optimize_hyperparameters(epochs=30, lr=0.0001)

        predictions = gp.predict(x_test)
        y_pred = predictions.mean.numpy()

        passed = np.all(np.abs(y_test - y_pred) < tol)

        if passed:
            self.assertTrue(True)
        else:
            print(f'Max(diff): {np.abs(y_test - y_pred).max()}')
            x_train = x_train.reshape(-1)
            x_test = x_test.reshape(-1)
            f, ax = plt.subplots(1, 1, figsize=(8,8))
            lower, upper = predictions.confidence_region()
            ax.plot(x_train, y_train, 'k*')
            ax.plot(x_test, y_pred, 'g-*')
            ax.plot(x_test, y_test, 'r*')
            ax.fill_between(x_test, lower.numpy(), upper.numpy(), alpha=0.5)
            ax.legend(['Observed Data', 'Prediction', 'Hidden data', 'Confidence','Noise'])
            ax.grid(True)
            plt.show()
            self.assertTrue(False)

    def test_hyperparameters_2(self):
        class SineGP(GP):
            @tensorwrap('train_x', 'train_y')
            def __init__(self, train_x, train_y):
                mean_module = gpytorch.means.ZeroMean()
                covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.CosineKernel()
                )
                likelihood = gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=constraint_from_tuple((1e-7, 1e-3))
                )
                init_dict = {
                    'covar_module.base_kernel.period_length': 1.
                }
                super(SineGP, self).__init__(train_x, train_y, mean_module, covar_module, likelihood)
                self.initialize(**init_dict)

        period_length = 2
        tol = 2e-4
        x = np.linspace(0,4,500).reshape((-1,1))
        y = np.cos(x * np.pi / period_length).reshape(-1)
        x_train = x[:250]
        y_train = y[:250]
        x_test = x[250:]
        y_test = y[250:]

        # Contrary to test_hyperparameters_1, the Cosine kernel GP should get the same period_length than the one
        # used to generate the data
        gp = SineGP(x_train, y_train)
        gp.optimizer = torch.optim.Adam
        gp.optimize_hyperparameters(epochs=30, lr=.1)
        gp.optimize_hyperparameters(epochs=20, lr=0.01)
        gp.optimize_hyperparameters(epochs=30, lr=0.001)

        predictions = gp.predict(x_test)
        y_pred = predictions.mean.numpy()

        passed = np.all(np.abs(y_test - y_pred) < tol)
        passed &= np.abs(gp.covar_module.base_kernel.period_length.item() - period_length) < tol

        if passed:
            self.assertTrue(True)
        else:
            print(f'Period length: predicted= {gp.covar_module.base_kernel.period_length.item()} | true= {period_length}')
            print(f'Max(diff): {np.abs(y_test - y_pred).max()}')
            x_train = x_train.reshape(-1)
            x_test = x_test.reshape(-1)
            f, ax = plt.subplots(1, 1, figsize=(8, 8))
            lower, upper = predictions.confidence_region()
            ax.plot(x_train, y_train, 'k*')
            ax.plot(x_test, y_pred, 'b')
            ax.plot(x_test, y_test, 'r*')
            ax.fill_between(x_test, lower.numpy(), upper.numpy(), alpha=0.5)
            ax.legend(['Observed Data', 'Prediction', 'Hidden data', 'Confidence','Noise'])
            ax.grid(True)
            plt.show()
            self.assertTrue(False)


    def test_data_manipulation(self):
        tol = 1e-1
        x = np.linspace(0, 1, 101).reshape((-1, 1))
        y = np.exp(-x**2).reshape(-1)
        x_ = np.linspace(1.5, 2, 51).reshape((-1, 1))
        y_ = 1 + np.exp(-x_**2).reshape(-1)

        gp = MaternGP(x, y, noise_prior=(0.1, 0.1))

        tmp = gp.empty_data()
        self.assertEqual(tmp, gp)
        self.assertTrue(tuple(gp.train_x.shape), (0, 1))
        # GPyTorch fails when predicting with an empty dataset, so the following line fails if uncommented
        # gp.predict(x)

        gp.set_data(x, y)
        self.assertEqual(tuple(gp.train_x.shape), (len(x), 1))

        gp.optimize_hyperparameters(epochs=10)
        gp_pred = gp.predict(x_).mean.numpy()
        self.assertFalse(np.all(np.abs(gp_pred - y_) < tol))

        tmp = gp.append_data(x_, y_)
        # self.assertTrue(gp != tmp)
        # self.assertEqual(tuple(gp.train_x.shape), (len(x), 1))
        self.assertEqual(tuple(tmp.train_x.shape), (len(x) + len(x_), 1))

        tmp.optimize_hyperparameters(epochs=10)
        tmp_pred = tmp.predict(x_).mean.numpy()
        self.assertTrue(np.all(np.abs(tmp_pred - y_) < tol))

    def test_multivariate_normal_prior(self):
        x = np.linspace(0, 1, 100, dtype=np.float32).reshape((-1, 2))
        y = np.exp(-x @ x.T).reshape(-1)

        gp = MaternGP(
            x, y,
            lengthscale_prior=((1, 0.01), (10, 1))
        )
        kernel = gp.covar_module.base_kernel
        prior = kernel.lengthscale_prior
        self.assertIsInstance(gp.covar_module.base_kernel.lengthscale_prior,
                              gpytorch.priors.MultivariateNormalPrior)
        self.assertTrue((kernel.lengthscale.squeeze() == prior.mean).all())

    def test_timeforgetting_dataset(self):
        x = np.linspace(0, 1, 100, dtype=np.float32).reshape((-1, 1))
        y = np.exp(-x**2).reshape(-1)

        gp = MaternGP(
            x, y, noise_constraint=(0,1e-3), dataset_type='timeforgetting', dataset_params={'keep': 50}
        )
        self.assertTrue((gp.train_x.numpy() == x[-50:]).all())
        self.assertTrue((gp.train_y.numpy() == y[-50:]).all())
        gp.append_data(x[:10], y[:10])
        self.assertTrue((gp.train_x.numpy() == np.vstack((x[-40:], x[:10]))).all())
        self.assertTrue((gp.train_y.numpy() == np.hstack((y[-40:], y[:10]))).all())
        gp.set_data(x[:75], y[:75])
        self.assertTrue((gp.train_x.numpy() == x[25:75]).all())
        self.assertTrue((gp.train_y.numpy() == y[25:75]).all())

    def test_neighbor_erasing_dataset_0(self):
        x = np.linspace(0, 1, 10).reshape((-1, 1))
        y = np.sin(x).squeeze()

        x_ = np.linspace(0, 1, 5).reshape((-1, 1))
        y_ = np.sin(x_).squeeze()

        distances = np.abs(x - x_.reshape(-1)).min(axis=1)

        radius = 1.1 * np.min(distances[np.nonzero(distances)])

        x_are_close = [(np.abs(xx - x_) < radius).any() for xx in x.squeeze()]
        x_kept = x[np.logical_not(x_are_close)]
        y_kept = y[np.logical_not(x_are_close)]
        x_final = np.concatenate((x_kept, x_), axis=0)
        y_final = np.concatenate((y_kept, y_), axis=-1)

        x = ensure_tensor(x)
        y = ensure_tensor(y)
        x_ = ensure_tensor(x_)
        y_ = ensure_tensor(y_)

        ds = NeighborErasingDataset(x, y, radius)

        ds.append(x_, y_)

        self.assertTrue(np.isclose(ds.train_x.numpy(), x_final).all())
        self.assertTrue(np.isclose(ds.train_y.numpy(), y_final).all())

    def test_neighbor_erasing_dataset_1(self):
        x = np.linspace(0, 1, 100, dtype=np.float32).reshape((-1, 1))
        y = np.exp(-x ** 2).reshape(-1)

        r = 0.0125

        gp = MaternGP(
            x, y, noise_constraint=(0, 1e-3), dataset_type='neighborerasing',
            dataset_params={'radius': r}
        )
        x_ = np.linspace(0, 1, 20, dtype=np.float32).reshape((-1, 1))
        y_ = np.exp(-x_ ** 2).reshape(-1)

        gp.append_data(x_, y_)
        distances = np.abs(gp.train_x.numpy().reshape((-1, 1)) - x_.squeeze())

        self.assertTrue(np.all(
            distances[:-len(x_), :] >= r
        ))
        self.assertTrue((gp.train_x.numpy()[-len(x_):, :] == x_).all())
        # Set to True to plot
        if False:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.scatter(
                gp.train_x.numpy()[-len(x_):],
                gp.train_x.numpy()[-len(x_):],
                color='r',
            )
            plt.scatter(
                gp.train_x.numpy()[:-len(x_)],
                gp.train_x.numpy()[:-len(x_)],
                color='b'
            )
            plt.show()

    def test_neighbor_erasing_dataset_2(self):
        x = np.linspace(0, 1, 100, dtype=np.float32).reshape((-1, 1))
        y = np.exp(-x ** 2).reshape(-1)

        r = 0.0125

        gp = MaternGP(
            x, y, noise_constraint=(0, 1e-3), dataset_type='neighborerasing',
            dataset_params={'radius': r}
        )
        x_ = np.linspace(0, 1, 20, dtype=np.float32).reshape((-1, 1))
        y_ = np.exp(-x_ ** 2).reshape(-1)
        forgettable = [False] * len(y_)

        gp.append_data(x_, y_, forgettable=forgettable)

        x__ = np.linspace(0, 1, 21, dtype=np.float32).reshape((-1, 1))
        y__ = np.exp(-x_ ** 2).reshape(-1)

        gp.append_data(x__, y__)

        all_present_x_ = all(x_ex.tolist() in gp.train_x.numpy().tolist()
                             for x_ex in x_)
        all_present_x__ = all(x__ex.tolist() in gp.train_x.numpy().tolist()
                              for x__ex in x__)

        self.assertTrue(all_present_x_ and all_present_x__)

    def test_neighbor_erasing_dataset_3(self):
        x = np.linspace(0, 1, 100, dtype=np.float32).reshape((-1, 1))
        y = np.exp(-x ** 2).reshape(-1)

        r = 0.0125

        gp = MaternGP(
            x, y, noise_constraint=(0, 1e-3), dataset_type='neighborerasing',
            dataset_params={'radius': r}
        )
        x_ = np.linspace(0, 1, 20, dtype=np.float32).reshape((-1, 1))
        y_ = np.exp(-x_ ** 2).reshape(-1)
        make_forget = [False] * len(y_)

        gp.append_data(x_, y_, make_forget=make_forget)

        all_present_x = all(x_ex.tolist() in gp.train_x.numpy().tolist()
                             for x_ex in x)
        all_present_x_ = all(x_ex.tolist() in gp.train_x.numpy().tolist()
                              for x_ex in x_)

        self.assertTrue(all_present_x and all_present_x_)

    def test_multi_dim_input(self):
        tol = 0.1
        lin = np.linspace(0, 1, 101)
        x = lin.reshape((-1, 1))
        x = x + x.T
        y = np.exp(-lin**2)

        gp = MaternGP(x, y, noise_prior=(0.1, 0.1))
        gp.optimize_hyperparameters(epochs=10)

        x_query = np.linspace(0.25, 0.75, 27)
        y_ = np.exp(-x_query**2)
        x_query = x_query.reshape((-1, 1)) + lin.reshape((1, -1))

        pred = gp.predict(x_query).mean.numpy()
        self.assertEqual(pred.shape, (27,))
        self.assertTrue(np.all(np.abs(pred - y_) < tol))

