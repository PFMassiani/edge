import numpy as np


class GaussianDensity:
    def __init__(self, output_dim, features_function=None, n_features=None, initial_weight=0, initial_var=1):
        if n_features is None and (not isinstance(initial_weight, np.ndarray) or initial_weight.ndim != 2):
            raise ValueError('Please either specify n_features or give a 2d-array for initial_weight')
        elif n_features is None:
            n_features = initial_weight.shape[1]

        if not isinstance(initial_weight, np.ndarray):
            initial_weight = np.ones((output_dim, n_features), dtype=float) * initial_weight
        if not isinstance(initial_var, np.ndarray):
            initial_var = np.ones(output_dim, dtype=float) * initial_var

        if features_function is None:
            def identity(x):
                return x
            features_function = identity

        self.n_features = n_features
        self.output_dim = output_dim
        self.weight = initial_weight
        self.var = initial_var
        self.features_function = features_function

    def __call__(self, x):
        features = self.features_function(x)
        # If x is 2d, we transpose the features, and if x is 1d, the features are too, so transposing does nothing
        mean = self.weight @ features.T
        density = np.random.normal(mean, self.var)
        return density

    def gradient_of_log(self, x):
        features = self.features_function(x)
        mean = self.weight @ features.T

        def grad_func(a):
            return ((a - mean) / self.var) * features.T
        return grad_func

    def update(self, step):
        self.weight += step

    def __str__(self):
        return f'GaussianDensity(output_dim={self.output_dim}, features_function={str(self.features_function)}, ' \
               f'n_features={self.n_features}, weight={self.weight}, var={self.var})'