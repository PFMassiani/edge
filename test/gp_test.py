import unittest
import torch
import numpy as np
import tempfile
import os
import warnings
import gpytorch

from edge.model.inference import tensorwrap
from edge.model.inference import MaternGP


class TestTensorwrap(unittest.TestCase):
    def check_tensor(self, *args, **kwargs):
        for a in args:
            self.assertTrue(torch.is_tensor(a))
        for k, a in kwargs.items():
            self.assertTrue(torch.is_tensor(a))

    def test_for_method(self):
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

    def test_for_class(me):
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


class TestMaternGP(unittest.TestCase):
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

    def test_hyper_optimization(self):
        warnings.simplefilter('ignore', gpytorch.utils.warnings.GPInputWarning)

        tol = 5e-2
        x = np.linspace(0, 1, 101).reshape((-1, 1))
        y = np.exp(-x**2).reshape(-1)

        gp = MaternGP(x, y, noise_prior=(0.1, 0.1))
        gp.optimize_hyperparameters(epochs=50)

        predictions = gp.predict(x).mean.numpy()

        self.assertTrue(np.all(np.abs(y - predictions) < tol))

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
