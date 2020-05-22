import numpy as np
from scipy.stats import norm
from pathlib import Path

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
            return mean, prediction.variance.detach().numpy()
        else:
            return mean

    def measure(self, state, lambda_threshold=0, gamma_threshold=None):
        if gamma_threshold is None:
            gamma_threshold = self.gamma_measure

        level_set = self.level_set(state, lambda_threshold, gamma_threshold,
                                   return_proba=False, return_covar=False)

        level_set = level_set.reshape((-1,) + self.env.action_space.shape)
        mean_axes = tuple([1 + k
                          for k in range(self.env.action_space.index_dim)])

        return np.atleast_1d(level_set.mean(mean_axes))

    def level_set(self, state, lambda_threshold, gamma_threshold,
                  return_proba=False, return_covar=False, return_measure=False):
        # We permit calling this function on different lambda and gamma
        # to avoid multiple inference passes
        if not isinstance(lambda_threshold, (list, tuple)):
            lambda_threshold_list = [lambda_threshold]
        else:
            lambda_threshold_list = lambda_threshold
        if not isinstance(gamma_threshold, (list, tuple)):
            gamma_threshold_list = [gamma_threshold]
        else:
            gamma_threshold_list = gamma_threshold

        if state is None:
            state = slice(None, None, None)
        action = slice(None, None, None)

        measure_slice, covar_slice = self.query(
            (state, action), return_covar=True)
        level_value_list = [
            norm.cdf(
                (measure_slice - lambda_threshold) / np.sqrt(covar_slice)
            )
            for lambda_threshold in lambda_threshold_list
        ]

        level_set_list = [level_value > gamma_threshold
                          for level_value, gamma_threshold in
                          zip(level_value_list, gamma_threshold_list)]

        if len(level_set_list) == 1:
            level_set_list = level_set_list[0]
            level_value_list = level_value_list[0]

        return_var = level_set_list
        if return_proba:
            return_var = (return_var, level_value_list)
        if return_covar:
            if not isinstance(return_var, tuple):
                return_var = (return_var, )
            return_var += (covar_slice, )
        if return_measure:
            if not isinstance(return_var, tuple):
                return_var = (return_var, )
            return_var += (measure_slice, )

        return return_var


class MaternSafety(SafetyMeasure):
    def __init__(self, env, gamma_measure, x_seed, y_seed, gp_params=None):
        if gp_params is None:
            gp_params = {}
        gp = MaternGP(x_seed, y_seed, **gp_params)
        super(MaternSafety, self).__init__(env, gp, gamma_measure)

    @staticmethod
    def load(load_folder, env, gamma_measure, x_seed, y_seed):
        load_path = Path(load_folder)
        gp_load_path = str(load_path / GPModel.GP_SAVE_NAME)

        gp = MaternGP.load(gp_load_path, x_seed, y_seed)

        model = MaternSafety(env, gamma_measure=gamma_measure,
                             x_seed=x_seed, y_seed=y_seed)
        model.gp = gp

        return model
