import numpy as np

from . import Agent
from edge.model.policy_models import ConstrainedEpsilonGreedy, SafetyMaximization
from edge.model.value_models import GPQLearning
from edge.model.safety_models import MaternSafety


class ValueAndSafetyLearner(Agent):
    """
    Defines an Agent modelling the Q-Values with a MaternGP updated with the Q-Learning update. The Agent also has a
    model for the underlying safety measure with a SafetyModel, and the Q-Learning update is constrained to stay in the
    current estimate of the safe set.
    """
    def __init__(self, env,
                 greed, step_size, discount_rate, q_x_seed, q_y_seed,
                 gamma_optimistic, gamma_cautious, lambda_cautious, s_x_seed, s_y_seed,
                 q_gp_params=None, s_gp_params=None, keep_seed_in_data=True):
        """
        Initializer
        :param env: the environment
        :param greed: the epsilon parameter of the ConstrainedEpsilonGreedy policy
        :param q_step_size: the step size in the Q-Learning update
        :param discount_rate: the discount rate
        :param q_x_seed: the seed input of the GP for the Q-Values model
        :param q_y_seed: the seed output of the GP for the Q-Values model
        :param gamma_optimistic: the gamma parameter for Q_optimistic
        :param gamma_cautious: the gamma parameter for Q_cautious
        :param lambda_cautious: the lambda parameter for Q_cautious
        :param s_x_seed: the seed input of the GP for the safety model
        :param s_y_seed: the seed output of the GP for the safety model
        :param q_gp_params: the parameters defining the GP for the Q-Values model. See edge.models.inference.MaternGP
            for more information
        :param q_gp_params: the parameters defining the GP for the safety model. See edge.models.inference.MaternGP
            for more information
        :param keep_seed_in_data: whether to keep the seed data in the GPs datasets. Should be True, otherwise GPyTorch
            fails.
        """
        Q_model = GPQLearning(env, step_size, discount_rate,
                              x_seed=q_x_seed, y_seed=q_y_seed,
                              gp_params=q_gp_params)
        safety_model = MaternSafety(env, gamma_optimistic,
                                    x_seed=s_x_seed, y_seed=s_y_seed,
                                    gp_params=s_gp_params)
        super(ValueAndSafetyLearner, self).__init__(env, Q_model, safety_model)

        self.Q_model = Q_model
        self.safety_model = safety_model
        self.lambda_cautious = lambda_cautious
        self.gamma_cautious = gamma_cautious
        self._gamma_optimistic = gamma_optimistic

        self.constrained_value_policy = ConstrainedEpsilonGreedy(
            self.env.stateaction_space, greed)
        self.safety_maximization_policy = SafetyMaximization(
            self.env.stateaction_space)

        self.keep_seed_in_data = keep_seed_in_data
        if not keep_seed_in_data:
            self.Q_model.empty_data()

    @property
    def greed(self):
        """
        Returns the epsilon parameter of the ConstrainedEpsilonGreedy policy
        :return: epsilon parameter
        """
        return self.constrained_value_policy.greed

    @greed.setter
    def greed(self, new_greed):
        """
        Sets the epsilon parameter of the ConstrainedEpsilonGreedy policy
        """
        self.constrained_value_policy.greed = new_greed

    @property
    def gamma_optimistic(self):
        return self._gamma_optimistic

    @gamma_optimistic.setter
    def gamma_optimistic(self, new_gamma_optimistic):
        self._gamma_optimistic = new_gamma_optimistic
        self.safety_model.gamma_measure = new_gamma_optimistic

    def get_next_action(self):
        is_cautious, proba_slice = self.safety_model.level_set(
            self.state,
            lambda_threshold=self.lambda_cautious,
            gamma_threshold=self.gamma_cautious,
            return_proba=True,
            return_covar=False
        )
        is_cautious = is_cautious.squeeze()
        proba_slice = proba_slice.squeeze()
        q_values = self.Q_model[self.state, :]

        action = self.constrained_value_policy.get_action(
            q_values, is_cautious
        )
        if action is None:
            print('No viable action, taking the safest...')
            action = self.safety_maximization_policy.get_action(proba_slice)

        return action

    def get_random_safe_state(self):
        """
        Returns a random state that is classified as safe by the safety model
        :return: a safe state
        """
        is_viable = self.safety_model.measure(
            slice(None, None, None),
            lambda_threshold=self.lambda_cautious,
            gamma_threshold=self.gamma_cautious
        ) > 0
        viable_indexes = np.atleast_1d(np.argwhere(is_viable).squeeze())
        try:
            state_index = viable_indexes[np.random.choice(len(viable_indexes))]
        except Exception as e:
            print('ERROR:', str(e))
        state = self.env.stateaction_space.state_space[state_index]
        return state

    def update_models(self, state, action, next_state, reward, failed, done):
        self.Q_model.update(state, action, next_state, reward, failed)
        self.safety_model.update(state, action, next_state, reward, failed)

    def fit_models(self, q_train_x=None, q_train_y=None, q_epochs=None, q_optimizer_kwargs=None,
                   s_train_x=None, s_train_y=None, s_epochs=None, s_optimizer_kwargs=None):
        if q_train_x is not None:
            if q_optimizer_kwargs is None:
                q_optimizer_kwargs = {}
            self.Q_model.fit(q_train_x, q_train_y, q_epochs, **q_optimizer_kwargs)
            if not self.keep_seed_in_data:
                self.Q_model.empty_data()

        if s_train_x is not None:
            if s_optimizer_kwargs is None:
                s_optimizer_kwargs = {}
            self.safety_model.fit(s_train_x, s_train_y, s_epochs, **s_optimizer_kwargs)
            if not self.keep_seed_in_data:
                self.safety_model.empty_data()
