import numpy as np
from scipy.stats import norm

from .. import GPModel
from ..inference import MaternGP


class SafetyMeasure(GPModel):
    def __init__(self, env, gp):
        super(SafetyMeasure, self).__init__(env, gp)

    def update(self, state, action, new_state, reward, failed):
        if not failed:
            update_value = self.measure(new_state)
        else:
            update_value = np.array([0.])

        stateaction = self.env.stateaction_space[state, action]
        self.gp.append_data(stateaction, update_value)

    def _query(self, x, return_covar=False):
        prediction = self.gp.predict(x)
        mean = prediction.mean.numpy()
        if return_covar:
            return mean, prediction.variance.numpy()
        else:
            return mean

    def measure(self, state):
        measure_slice = self[state, :]
        return np.atleast_1d(measure_slice.mean(axis=-1))

    def level_set(self, state, lambda_threshold, gamma_threshold,
                  return_covar=False):
        query = self.env.stateaction_space[state, :]

        measure_slice, covar_slice = self._query(query, return_covar=True)
        level_value = norm.cdf(
            (measure_slice - lambda_threshold) / np.sqrt(covar_slice)
        )

        level_set = level_value > gamma_threshold
        if not return_covar:
            return level_set
        else:
            return level_set, covar_slice


class MaternSafety(SafetyMeasure):
    def __init__(self, env, x_seed, y_seed, *gp_args, **gp_kwargs):
        gp = MaternGP(x_seed, y_seed, *gp_args, **gp_kwargs)
        super(MaternSafety, self).__init__(env, gp)
