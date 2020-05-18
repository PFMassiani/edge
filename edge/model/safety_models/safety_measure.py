import numpy as np
from scipy.stats import norm

from .. import GPModel
from ..inference import MaternGP


class SafetyMeasure(GPModel):
    def __init__(self, env, gp, gamma_measure):
        super(SafetyMeasure, self).__init__(env, gp)
        self.gamma_measure = gamma_measure

    def update(self, state, action, new_state, reward, failed):
        if not failed:
            update_value = self.measure(
                state=new_state,
                lambda_threshold=0,
                gamma_threshold=self.gamma_measure
            )
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

    def measure(self, state, lambda_threshold, gamma_threshold):
        level_set = self.level_set(state, lambda_threshold, gamma_threshold,
                                   return_proba=False, return_covar=False)

        level_set = level_set.reshape((-1,) + self.env.action_space.shape)
        mean_axes = tuple([1 + k
                          for k in range(self.env.action_space.index_dim)])

        return np.atleast_1d(level_set.mean(mean_axes))

    def level_set(self, state, lambda_threshold, gamma_threshold,
                  return_proba=False, return_covar=False):
        if state is None:
            state = slice(None, None, None)
        action = slice(None, None, None)

        measure_slice, covar_slice = self.query(
            (state, action), return_covar=True)
        level_value = norm.cdf(
            (measure_slice - lambda_threshold) / np.sqrt(covar_slice)
        )

        level_set = level_value > gamma_threshold

        return_var = level_set
        if return_proba:
            return_var = (return_var, level_value)
        if return_covar:
            if not isinstance(return_var, tuple):
                return_var = (return_var, )
            return_var += (covar_slice, )

        return return_var


class MaternSafety(SafetyMeasure):
    def __init__(self, env, gamma_optimistic, x_seed, y_seed, gp_params=None):
        if gp_params is None:
            gp_params = {}
        gp = MaternGP(x_seed, y_seed, **gp_params)
        super(MaternSafety, self).__init__(env, gp, gamma_optimistic)
