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
        # TODO correct this so it averages the indicator function
        measure_slice = self[state, :]
        actions_axes = list(range(
            self.env.state_space.index_dim,
            self.env.stateaction_space.index_dim
        ))
        return np.atleast_1d(measure_slice.mean(axis=actions_axes))

    def level_set(self, state, lambda_threshold, gamma_threshold,
                  return_proba=False, return_covar=False):
        query = self.env.stateaction_space[state, :]

        measure_slice, covar_slice = self._query(query, return_covar=True)
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

    def full_level_set(self, lambdas_threshold, gammas_threshold,
                       return_covar=False):
        if not isinstance(lambdas_threshold, (list, tuple)):
            lambdas_threshold = [lambdas_threshold]
        if not isinstance(gammas_threshold, (list, tuple)):
            gammas_threshold = [gammas_threshold]
        n_samples = len(lambdas_threshold)

        measure, covar = self.query(
            slice(None, None, None),
            return_covar=True
        )

        level_values = [norm.cdf((measure - l_threshold) / np.sqrt(covar))
                        for l_threshold in lambdas_threshold]

        level_sets = [level_value > gamma
                      for level_value, gamma in zip(level_values,
                                                    gammas_threshold)]

        if n_samples == 1:
            level_sets = level_sets[0]

        if return_covar:
            return level_sets, covar
        else:
            return level_sets


class MaternSafety(SafetyMeasure):
    def __init__(self, env, x_seed, y_seed, gp_params=None):
        if gp_params is None:
            gp_params = {}
        gp = MaternGP(x_seed, y_seed, **gp_params)
        super(MaternSafety, self).__init__(env, gp)
