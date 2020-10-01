import unittest
import numpy as np
import math

from edge.model.inference.custom_kernels_gps import SymmetricMaternCosGP


def get_gp(x, y):
    return SymmetricMaternCosGP(
        x, y,
        noise_prior=(1, 0.1),
        noise_constraint=(1e-3, 1e4),
        lengthscale_prior=(1.5, 0.1),
        lengthscale_constraint=(1e-3, 10),
        outputscale_prior=(1, 0.1),
        outputscale_constraint=(1e-3, 1e2),
        hyperparameters_initialization={
            'matern_0.lengthscale': 2
        },
        value_structure_discount_factor=0.5
    )


class SymmetricMaternCosGPTest(unittest.TestCase):
    def test_init(self):
        x = np.arange(12).reshape(-1,6)
        y = x.sum(axis=1)

        model = get_gp(x, y)
        self.assertTrue(
            (
                model.covar_module.base_kernel.base_kernel.kernels[0].
                kernels[0].lengthscale == 2
            ).all()
        )
        self.assertEqual(
            model.covar_module.base_kernel.base_kernel.kernels[0].\
            kernels[0].lengthscale.shape[1],
            4
        )
        self.assertEqual(
            model.covar_module.base_kernel.base_kernel.kernels[1].\
                period_length[0],
            math.pi
        )
        self.assertTrue(
            (
                (model.covar_module.base_kernel.base_kernel.kernels[2].
                 kernels[0].lengthscale - 1.5).abs() < 1e-6
            ).all()
        )
        self.assertEqual(
            model.covar_module.base_kernel.base_kernel.kernels[2].\
                kernels[0].lengthscale.shape[1],
            1
        )

    def test_forward(self):
        z = np.arange(18).reshape(-1, 6)
        x = z[(0, 1), :]
        x_test = z[2, :].reshape(1, 6)
        y = x.sum(axis=1)
        model = get_gp(x, y)

        y_test = model.forward(x_test)
        mean = y_test.mean
        covar = y_test.covariance_matrix

        self.assertEqual(mean.shape[0], 1)
        self.assertEqual(tuple(covar.shape), (1, 1))