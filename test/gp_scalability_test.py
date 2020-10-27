import unittest
import numpy as np
from edge.model.inference import MaternGP


DEBUG = False


class ScalabilityTests(unittest.TestCase):
        def test_large_ds_matern(self):
            x = np.linspace(0, 1, 9000, dtype=np.float32).reshape((-1, 1))
            y = np.exp(-x ** 2).reshape(-1)
            gp = MaternGP(
                x, y, noise_constraint=(0, 1e-3)
            )  
            x_ = np.linspace(0, 1, 20, dtype=np.float32).reshape((-1, 1))
            y_ = np.exp(-x_ ** 2).reshape(-1)
 
            try:
                pred = gp.predict(x_).mean.cpu().numpy()
            except Exception as e:
                if DEBUG:
                    import pudb
                    pudb.post_mortem()
                self.assertTrue(False, f'Prediction failed with the following error: {str(e)}')
            self.assertTrue(True)


        def test_large_ds_value_matern(self):
            x = np.linspace(0, 1, 9000, dtype=np.float32).reshape((-1, 1))
            y = np.exp(-x ** 2).reshape(-1)
            gp = MaternGP(
                x, y, noise_constraint=(0, 1e-3), value_structure_discount_factor=0.5
            )  
            x_ = np.linspace(0, 1, 20, dtype=np.float32).reshape((-1, 1))
            y_ = np.exp(-x_ ** 2).reshape(-1)
 
            try:
                pred = gp.predict(x_).mean.cpu().numpy()
            except Exception as e:
                if DEBUG:
                    import pudb
                    pudb.post_mortem()
                self.assertTrue(False, f'Prediction failed with the following error: {str(e)}')
            self.assertTrue(True)


if __name__ == '__main__':
        unittest.main()
