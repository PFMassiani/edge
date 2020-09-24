import numpy as np
from scipy.stats import norm

from . import Policy


class BayesianPolicy(Policy):
    def __init__(self, stateaction_space):
        super(BayesianPolicy, self).__init__(stateaction_space)

    def acquisition_function(self, mean, covar):
        """
        Acquisition function of the bayesian policy.
        If action is None, this function should return the value of the function
        on the whole action space
        :param mean: mean values of the GP
        :param covar: variances of the points of the GP
        :return: the values of the acquisition function at the given points
        """
        raise NotImplementedError

    def proposed_action(self, mean, covar, constraints=None):
        """
        Returns the action chosen by the acquisition function maximization.
        :return: the action with the best acquisition value
        """
        if constraints is None:
            constraints = np.ones_like(mean, dtype=bool)
        elif not constraints.any():
            return None
        else:
            mean = mean[constraints]
            covar = covar[constraints]
        acquisition_values = self.acquisition_function(mean, covar)
        action_constrained_index = np.argmax(acquisition_values)
        action_flat_index = np.argwhere(constraints)[
            action_constrained_index
        ][0]
        action_index = np.unravel_index(
            action_flat_index,
            self.stateaction_space.action_space.shape
        )
        action = self.stateaction_space.action_space[action_index]
        return action

    def get_action(self, mean, covar, constraints=None):
        raise NotImplementedError


class ExpectedImprovementPolicy(BayesianPolicy):
    def __init__(self, stateaction_space, xi):
        super(ExpectedImprovementPolicy, self).__init__(stateaction_space)
        self.best_sample = None
        self.xi = xi

    def get_action(self, mean, covar, best_sample, constraints=None):
        self.best_sample = best_sample
        action = self.proposed_action(mean, covar, constraints)
        return action

    def acquisition_function(self, mean, covar):
        with np.errstate(divide='ignore'):
            improvement = mean - self.best_sample - self.xi
            Z = improvement / covar
            ei = improvement * norm.cdf(Z) + covar * norm.pdf(Z)
            ei[covar == 0] = 0
        return ei


class SafetyInformationMaximization(BayesianPolicy):
    def __init__(self, stateaction_space):
        super(SafetyInformationMaximization, self).__init__(stateaction_space)

    def acquisition_function(self, mean, covar):
        return np.abs(covar)

    def get_action(self, covar, constraints=None):
        # The mean does not matter here
        return self.proposed_action(np.zeros_like(covar), covar, constraints)