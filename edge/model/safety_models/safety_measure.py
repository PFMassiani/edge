import numpy as np
from scipy.stats import norm
from pathlib import Path
import warnings, contextlib

from .. import GPModel
from ..inference import MaternGP


class SafetyMeasure(GPModel):
    """
    Safety measure as described in "A learnable safety measure", by Heim, von Rohr, et al. (2019, CoRL).
    """
    def __init__(self, space, gp, gamma_measure):
        """
        Initializer
        :param space: the Q-space (state-action space) that the model lives in.
        :param gp: the underlying GP
        :param gamma_measure: the gamma coefficient used by the measure. It corresponds to gamma_optimistic

        Note: the measure has no information about the gamma_cautious parameter: it is a parameter of the policy,
        not the measure. Hence, we rename gamma_optimistic to gamma_measure, since it is the only gamma parameter of the
        measure.
        """
        super(SafetyMeasure, self).__init__(space, gp)
        self.gamma_measure = gamma_measure

    def update(self, state, action, new_state, reward, failed):
        """
        Updates the underlying GP with the measure computation update
        :param state: the previous state
        :param action: the action taken
        :param new_state: the new state
        :param reward: the reward incurred
        :param failed: whether the agent has failed
        """
        if not failed:
            update_value = self.measure(
                state=new_state,
                lambda_threshold=0,
                gamma_threshold=self.gamma_measure
            )
        else:
            update_value = np.array([0.])

        stateaction = self.space[state, action]
        self.gp.append_data(stateaction, update_value, forgettable=[not failed],
                            make_forget=[not failed])

    def _query(self, x, return_covar=False):
        """
        Calls the GP model on the passed list of points. The covariance can also be returned.
        :param x: np.ndarray: a list of stateactions where the GP should be evaluated
        :param return_covar: boolean: whether the covariance should be returned as well
        :return: the mean value of the GP at these points, and if return_covar=True, the covariance at these points
        """
        prediction = self.gp.predict(x)
        mean = prediction.mean.numpy()
        if return_covar:
            return mean, prediction.variance.detach().numpy()
        else:
            return mean

    def measure(self, state, lambda_threshold=0, gamma_threshold=None):
        """
        Computes the safety measure in the state passed as parameter with the given thresholds.
        :param state: np.ndarray: the state or list of states where to compute the measure
        :param lambda_threshold: (default=0) the lambda parameter
        :param gamma_threshold: (default=gamma_measure) the gamma parameter
        :return: the measure at the given state(s)
        """
        if gamma_threshold is None:
            gamma_threshold = self.gamma_measure

        level_set = self.level_set(state, lambda_threshold, gamma_threshold,
                                   return_proba=False, return_covar=False)

        level_set = level_set.reshape((-1,) + self.space.action_space.shape)
        mean_axes = tuple([1 + k
                          for k in range(self.space.action_space.index_dim)])

        return np.atleast_1d(level_set.mean(mean_axes))

    def level_set(self, state, lambda_threshold, gamma_threshold,
                  return_proba=False, return_covar=False, return_measure=False):
        """
        Computes the probabilistic level set of the GP on the stateaction space. The output is a boolean array which
        is True whenever the stateaction is within the level set.
        If you want to consider multiple lambda and gamma thresholds, calling this method with a list of thresholds
        is more efficient than calling it multiple times. Note that when calling it with lists, the two lists should
        have the same length.
        :param state: the state or list of state from where we compute the level set
        :param lambda_threshold: the value or list of values to consider for the lambda threshold
        :param gamma_threshold: the value or list of values to consider for the gamma threshold
        :param return_proba: whether to return the probabilities associated to each state-action having a measure
        higher than the threshold
        :param return_covar: whether to return the covariance at each state-action
        :param return_measure: whether to return the actual values of the measure instead of the level set only
        :return: depending on the values of return_[proba,covar,measure], either a single array or a tuple of arrays
        is returned. Moreover, if lambda_threshold and gamma_threshold are lists, each array is replaced by a list of
        the arrays corresponding to each value for the thresholds.
        The first three arrays have the same shape : (state.shape[0],) + action_space.shape
        The last one has shape (state.shape[0],)
            * np.ndarray<boolean>: level set
            * np.ndarray<float>: probability of each state-action being above the lambda threshold
            * np.ndarray<float>: covariance at each state-action
            * np.ndarray<float>: value of the measure at each state
        """
        # We allow calling this function on different lambdas and gammas
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
            # Unspecfied state means the whole state space
            state = slice(None, None, None)
        action = slice(None, None, None)
        output_shape = (-1, ) + self.space.action_space.shape

        measure_slice, covar_slice = self.query(
            tuple(self.space.get_stateaction(state, action)),
            return_covar=True
        )
        measure_slice = measure_slice.reshape(output_shape)
        covar_slice = covar_slice.reshape(output_shape)

        # The following prints a user-friendly warning if a runtime warning is encountered in the computation of
        # level_value_list
        # If the kernel matrix is ill-conditioned, the covariance may be negative
        # This will raise a RuntimeWarning in np.sqrt(covar_slice)
        # See https://github.com/cornellius-gp/gpytorch/issues/1037
        with warnings.catch_warnings(record=True) as w:
            # The contextmanager decorator enables the use of the function in a `with` statement, and
            # requires that the function is a generator
            # This function simply returns the list it computes
            @contextlib.contextmanager
            def compute_cdf():
                try:
                    yield [
                        norm.cdf(
                            (measure_slice - lambda_threshold) / np.sqrt(covar_slice)
                        ) for lambda_threshold in lambda_threshold_list
                    ]
                finally:
                    pass
            with compute_cdf() as cdf_list:  # The list computed by compute_cdf is stored in cdf_list
                if len(w) > 0:  # We check whether a warning was raised to change its message
                    original_warning = ''
                    for wrng in w:
                        original_warning += str(wrng.message) + '\n'
                    original_warning = original_warning[:-2]
                    warning_message = ('Warning encountered in cumulative density function computation. \nThis may be '
                                       'caused by an ill-conditioned kernel matrix causing a negative covariance.\n'
                                       'Original warning: ' + str(original_warning))
                    level_value_list = [
                        norm.cdf(
                            (measure_slice - lambda_threshold) / np.sqrt(np.abs(covar_slice))
                        ) for lambda_threshold in lambda_threshold_list
                    ]
                else:
                    warning_message = None
                    level_value_list = cdf_list  # We store cdf_list so its value is available outside of `with`
        if warning_message is not None:
            warnings.warn(warning_message)

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
    def __init__(self, space, gamma_measure, x_seed, y_seed, gp_params=None):
        """
        Initializer
        :param space: the Q-space (state-action space) that the model lives in.
        :param gamma_measure: the gamma coefficient used by the measure. It corresponds to gamma_optimistic
        :param x_seed: the seed input of the GP: a list of stateactions
        :param y_seed: the seed output of the GP: a list of floats
        :param gp_params: the parameters defining the GP. See edge.models.inference.MaternGP for more information
        """
        if gp_params is None:
            gp_params = {}
        gp = MaternGP(x_seed, y_seed, **gp_params)
        super(MaternSafety, self).__init__(space, gp, gamma_measure)

    @staticmethod
    def load(load_folder, space, gamma_measure, x_seed, y_seed):
        """
        Loads the model and the GP saved by the GPModel.save method. Note that this method may fail if the save was
        made with an older version of the code.
        :param load_folder: the folder where the files are
        :return: MaternSafety: the model
        """
        load_path = Path(load_folder)
        gp_load_path = str(load_path / GPModel.GP_SAVE_NAME)

        gp = MaternGP.load(gp_load_path, x_seed, y_seed)

        model = MaternSafety(space, gamma_measure=gamma_measure,
                             x_seed=x_seed, y_seed=y_seed)
        model.gp = gp

        return model
