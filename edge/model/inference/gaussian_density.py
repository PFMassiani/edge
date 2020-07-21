import numpy as np


class GaussianDensity:
    def __init__(self, features_function=None, n_features=None, initial_mean=0, initial_var=1):
        if n_features is None and (not isinstance(initial_mean, np.ndarray) or initial_mean.ndim != 2):
            raise ValueError('Please either specify n_features or give a 2d-array for initial_mean')
        else:
            n_features = initial_mean.shape[1]

        if not isinstance(initial_mean, np.ndarray):
            initial_mean = np.ones((n_features, n_features), dtype=float) * initial_mean
        if not isinstance(initial_var, np.ndarray):
            initial_var = np.ones(n_features, dtype=float) * initial_var

        if features_function is None:
            def identity(x):
                return x
            features_function = identity

        self.n_features = n_features
        self.mean = initial_mean
        self.var = initial_var
        self.features_function = features_function

    def __call__(self, x):
        features = self.features_function(x)
        # If x is 2d, we transpose the features, and if x is 1d, the features are too, so transposing does nothing
        mean = self.mean @ features.T
        density = np.random.normal(mean, self.var)
        return density

    def gradient_of_log(self, x):
        features = self.features_function(x)
        mean = self.mean @ features

        def grad_func(a):
            return ((a - mean) / self.var) @ features.T
        return grad_func

    def update(self, step):
        self.mean += step