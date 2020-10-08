import numpy as np
import logging

logger = logging.getLogger(__name__)

from edge.agent import QLearner, Agent
from edge.model.safety_models import MaternSafety
from edge.model.value_models import GPQLearning
from edge.model.policy_models import ConstrainedEpsilonGreedy, \
    SafetyMaximization, SafetyActiveSampling


def affine_interpolation(t, start, end):
    return start + (end - start) * t


class ValuesAndSafetyCombinator(QLearner):
    """
        Defines an Agent modelling the Q-Values with a MaternGP updated with the
         Q-Learning update. The Agent also has a model for the underlying safety
        measure with a SafetyModel, and the Q-Learning update is constrained to
        stay in the current estimate of the safe set. If the update chosen by
        Q-Learning is within a conservative estimate of the viable set, then the
        sample is used to update the safety model.
    """

    def __init__(self, env,
                 greed, step_size, discount_rate, q_x_seed, q_y_seed,
                 gamma_optimistic, gamma_cautious, lambda_cautious,
                 s_x_seed, s_y_seed, q_gp_params=None, s_gp_params=None,
                 keep_seed_in_data=True):
        """
        Initializer
        :param env: the environment
        :param greed: the epsilon parameter of the ConstrainedEpsilonGreedy
            policy
        :param step_size: the step size in the Q-Learning update
        :param discount_rate: the discount rate
        :param q_x_seed: the seed input of the GP for the Q-Values model
        :param q_y_seed: the seed output of the GP for the Q-Values model
        :param gamma_optimistic: the gamma parameter for Q_optimistic
        :param gamma_cautious: the gamma parameter for Q_cautious
        :param lambda_cautious: the lambda parameter for Q_cautious
        :param s_x_seed: the seed input of the GP for the safety model
        :param s_y_seed: the seed output of the GP for the safety model
        :param q_gp_params: the parameters defining the GP for the Q-Values
            model. See edge.models.inference.MaternGP
            for more information
        :param q_gp_params: the parameters defining the GP for the safety model.
            See edge.models.inference.MaternGP for more information
        :param keep_seed_in_data: whether to keep the seed data in the GPs
            datasets. Should be True, otherwise GPyTorch fails.
        """
        self.lambda_cautious_start, self.lambda_cautious_end = lambda_cautious
        self.gamma_cautious_start, self.gamma_cautious_end = gamma_cautious
        self.gamma_optimistic_start, self.gamma_optimistic_end = \
            gamma_optimistic
        self.lambda_cautious = self.lambda_cautious_start
        self.gamma_cautious = self.gamma_cautious_start

        self._step_size_decrease_index = 1

        Q_model = GPQLearning(env, step_size, discount_rate,
                              x_seed=q_x_seed, y_seed=q_y_seed,
                              gp_params=q_gp_params)
        safety_model = MaternSafety(env, self.gamma_optimistic_start,
                                    x_seed=s_x_seed, y_seed=s_y_seed,
                                    gp_params=s_gp_params)
        super(ValuesAndSafetyCombinator, self).__init__(
            env=env,
            greed=greed,  # Unused: we define another policy
            step_size=step_size,
            discount_rate=discount_rate,
            x_seed=q_x_seed,
            y_seed=q_y_seed,
            gp_params=q_gp_params,
            keep_seed_in_data=keep_seed_in_data
        )

        self.Q_model = Q_model
        self.safety_model = safety_model

        self.constrained_value_policy = ConstrainedEpsilonGreedy(
            self.env.stateaction_space, greed)
        self.safety_maximization_policy = SafetyMaximization(
            self.env.stateaction_space)
        self._training_greed = self.greed

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
        self._training_greed = new_greed

    @property
    def gamma_optimistic(self):
        return self.safety_model.gamma_measure

    @gamma_optimistic.setter
    def gamma_optimistic(self, new_gamma_optimistic):
        self.safety_model.gamma_measure = new_gamma_optimistic

    @Agent.training_mode.setter
    def training_mode(self, new_training_mode):
        if self.training_mode and not new_training_mode:  # From train to test
            self._training_greed = self.greed
            self.greed = 0
        elif (not self.training_mode) and new_training_mode:  # Test to train:
            self.greed = self._training_greed
        self._training_mode = new_training_mode

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
            logger.error('ERROR:', str(e))
            return None
        state = self.env.stateaction_space.state_space[state_index]
        return state

    def update_models(self, state, action, next_state, reward, failed, done):
        self.Q_model.update(state, action, next_state, reward, failed)
        self.safety_model.update(state, action, next_state, reward, failed)

    def fit_models(self, q_train_x=None, q_train_y=None, q_epochs=None,
                   q_optimizer_kwargs=None,
                   s_train_x=None, s_train_y=None, s_epochs=None,
                   s_optimizer_kwargs=None):
        if q_train_x is not None:
            if q_optimizer_kwargs is None:
                q_optimizer_kwargs = {}
            self.Q_model.fit(train_x=q_train_x, train_y=q_train_y,
                             epochs=q_epochs,
                             **q_optimizer_kwargs)
            if not self.keep_seed_in_data:
                self.Q_model.empty_data()

        if s_train_x is not None:
            if s_optimizer_kwargs is None:
                s_optimizer_kwargs = {}
            self.safety_model.fit(train_x=s_train_x, train_y=s_train_y,
                                  epochs=s_epochs,
                                  **s_optimizer_kwargs)
            if not self.keep_seed_in_data:
                self.safety_model.empty_data()

    def safety_parameters_affine_update(self, t):
        self.gamma_optimistic = affine_interpolation(
            t, self.gamma_optimistic_start, self.gamma_optimistic_end
        )
        self.gamma_cautious = affine_interpolation(
            t, self.gamma_cautious_start, self.gamma_cautious_end
        )
        self.lambda_cautious = affine_interpolation(
            t, self.lambda_cautious_start, self.lambda_cautious_end
        )


class SoftHardLearner(ValuesAndSafetyCombinator):
    def __init__(self, env,
                 greed, step_size, discount_rate, q_x_seed, q_y_seed,
                 gamma_optimistic, gamma_cautious, lambda_cautious, gamma_soft,
                 s_x_seed, s_y_seed, q_gp_params=None, s_gp_params=None,
                 keep_seed_in_data=True):
        super(SoftHardLearner, self).__init__(
            env=env,
            greed=greed,
            step_size=step_size,
            discount_rate=discount_rate,
            q_x_seed=q_x_seed,
            q_y_seed=q_y_seed,
            gamma_optimistic=gamma_optimistic,
            gamma_cautious=gamma_cautious,
            lambda_cautious=lambda_cautious,
            s_x_seed=s_x_seed,
            s_y_seed=s_y_seed,
            q_gp_params=q_gp_params,
            s_gp_params=s_gp_params,
            keep_seed_in_data=keep_seed_in_data
        )

        self.gamma_soft_start, self.gamma_soft_end = gamma_soft
        self.gamma_soft = self.gamma_soft_start

        self.violated_soft_constraint = None
        self.updated_safety = None

    def safety_parameters_affine_update(self, t):
        super(SoftHardLearner, self).safety_parameters_affine_update(t)
        self.gamma_soft = affine_interpolation(
            t, self.gamma_soft_start, self.gamma_soft_end
        )

    def get_next_action(self):
        is_cautious_list, proba_slice_list = self.safety_model.level_set(
            self.state,
            lambda_threshold=[self.lambda_cautious, self.lambda_cautious],
            gamma_threshold=[self.gamma_cautious, self.gamma_soft],
            return_proba=True,
            return_covar=False
        )
        is_cautious_hard, is_cautious_soft = is_cautious_list
        is_cautious_hard = is_cautious_hard.squeeze()
        is_cautious_soft = is_cautious_soft.squeeze()
        proba_slice_hard = proba_slice_list[0].squeeze()

        q_values = self.Q_model[self.state, :]
        action = self.constrained_value_policy.get_action(
            q_values, is_cautious_hard
        )

        if action is None:
            logger.info('No viable action, taking the safest...')
            action = self.safety_maximization_policy.get_action(
                proba_slice_hard)
            self.violated_soft_constraint = True
        else:
            action_idx = self.env.action_space.get_index_of(action)
            self.violated_soft_constraint = not is_cautious_soft[action_idx]

        return action

    def step(self):
        """
        Chooses an action according to the policy, takes a step in the
        Environment, and updates the models. The action taken is available in
        self.last_action.
        :return: new_state, reward, failed
        """
        old_state = self.state
        action = self.get_next_action()
        new_state, reward, failed = self.env.step(action)
        if self.training_mode:
            if self.violated_soft_constraint or failed:
                self.Q_model.update(old_state, action, new_state, reward,
                                    failed)
                self.safety_model.update(old_state, action, new_state, reward,
                                         failed)
                self.updated_safety = True
            else:
                self.Q_model.update(old_state, action, new_state, reward,
                                    failed)
                self.updated_safety = False
        else:
            self.updated_safety = False
        self.state = new_state
        self.last_action = action
        return new_state, reward, failed, self.env.done


class EpsilonSafety(ValuesAndSafetyCombinator):
    def __init__(self, env, epsilon,
                 greed, step_size, discount_rate, q_x_seed, q_y_seed,
                 gamma_optimistic, gamma_cautious, lambda_cautious,
                 s_x_seed, s_y_seed, q_gp_params=None, s_gp_params=None,
                 keep_seed_in_data=True):

        super(EpsilonSafety, self).__init__(
            env=env,
            greed=greed,
            step_size=step_size,
            discount_rate=discount_rate,
            q_x_seed=q_x_seed,
            q_y_seed=q_y_seed,
            gamma_optimistic=gamma_optimistic,
            gamma_cautious=gamma_cautious,
            lambda_cautious=lambda_cautious,
            s_x_seed=s_x_seed,
            s_y_seed=s_y_seed,
            q_gp_params=q_gp_params,
            s_gp_params=s_gp_params,
            keep_seed_in_data=keep_seed_in_data
        )

        self.active_sampling_policy = SafetyActiveSampling(
            self.env.stateaction_space
        )
        self.epsilon = epsilon
        self.explored_safety = False
        self.violated_constraint = False
        self.updated_safety = False

    def get_next_action(self):
        explore_safety = np.random.binomial(1, self.epsilon) == 1

        is_cautious, proba_slice, covar_slice = self.safety_model.level_set(
            self.state,
            lambda_threshold=self.lambda_cautious,
            gamma_threshold=self.gamma_cautious,
            return_proba=True,
            return_covar=True
        )
        is_cautious = is_cautious.squeeze()
        proba_slice = proba_slice.squeeze()
        covar_slice = covar_slice.squeeze()

        if explore_safety:
            action = self.active_sampling_policy.get_action(
                covar_slice, is_cautious
            )
            self.explored_safety = True
            self.violated_constraint = False
        else:
            q_values = self.Q_model[self.state, :]
            action = self.constrained_value_policy.get_action(
                q_values, is_cautious
            )
            self.explored_safety = False
            self.violated_constraint = False
        if action is None:
            logger.info('No viable action, taking the safest...')
            action = self.safety_maximization_policy.get_action(
                proba_slice
            )
            self.explored_safety = False
            self.violated_constraint = True

        return action

    def step(self):
        old_state = self.state
        action = self.get_next_action()
        new_state, reward, failed = self.env.step(action)
        if self.training_mode:
            self.Q_model.update(old_state, action, new_state, reward,
                                failed)
            if self.explored_safety or self.violated_constraint or failed:
                self.safety_model.update(old_state, action, new_state, reward,
                                         failed)
                self.updated_safety = True
            else:
                self.updated_safety = False
        else:
            self.updated_safety = False
        self.state = new_state
        self.last_action = action
        return new_state, reward, failed, self.env.done


class SafetyQLearningSwitcher(ValuesAndSafetyCombinator):
    def __init__(self, env,
                 greed, step_size, discount_rate, q_x_seed, q_y_seed,
                 gamma_optimistic, gamma_cautious, lambda_cautious,
                 s_x_seed, s_y_seed, q_gp_params=None, s_gp_params=None,
                 keep_seed_in_data=True):

        super(SafetyQLearningSwitcher, self).__init__(
            env=env,
            greed=greed,
            step_size=step_size,
            discount_rate=discount_rate,
            q_x_seed=q_x_seed,
            q_y_seed=q_y_seed,
            gamma_optimistic=gamma_optimistic,
            gamma_cautious=gamma_cautious,
            lambda_cautious=lambda_cautious,
            s_x_seed=s_x_seed,
            s_y_seed=s_y_seed,
            q_gp_params=q_gp_params,
            s_gp_params=s_gp_params,
            keep_seed_in_data=keep_seed_in_data
        )

        self.explore_safety = True
        self.active_sampling_policy = SafetyActiveSampling(
            self.env.stateaction_space
        )
        self.violated_constraint = False
        self.updated_safety = False

    def get_next_action(self):
        is_cautious, proba_slice, covar_slice = self.safety_model.level_set(
            self.state,
            lambda_threshold=self.lambda_cautious,
            gamma_threshold=self.gamma_cautious,
            return_proba=True,
            return_covar=True
        )
        is_cautious = is_cautious.squeeze()
        proba_slice = proba_slice.squeeze()
        covar_slice = covar_slice.squeeze()

        if self.explore_safety:
            action = self.active_sampling_policy.get_action(
                covar_slice, is_cautious
            )
            self.violated_constraint = False
        else:
            q_values = self.Q_model[self.state, :]
            action = self.constrained_value_policy.get_action(
                q_values, is_cautious
            )
            self.violated_constraint = False
        if action is None:
            logger.info('No viable action, taking the safest...')
            action = self.safety_maximization_policy.get_action(
                proba_slice
            )
            self.violated_constraint = True

        return action

    def step(self):
        old_state = self.state
        action = self.get_next_action()
        new_state, reward, failed = self.env.step(action)
        if self.training_mode:
            self.Q_model.update(old_state, action, new_state, reward,
                                failed)
            if self.explore_safety or self.violated_constraint or failed:
                self.safety_model.update(old_state, action, new_state, reward,
                                         failed)
                self.updated_safety = True
            else:
                self.updated_safety = False
        else:
            self.updated_safety = False
        self.state = new_state
        self.last_action = action
        return new_state, reward, failed, self.env.done


class SafetyQLearner(ValuesAndSafetyCombinator):
    def __init__(self, env,
                 greed, step_size, discount_rate, q_x_seed, q_y_seed,
                 gamma_optimistic, gamma_cautious, lambda_cautious,
                 s_x_seed, s_y_seed, q_gp_params=None, s_gp_params=None,
                 keep_seed_in_data=True):

        super(SafetyQLearner, self).__init__(
            env=env,
            greed=greed,
            step_size=step_size,
            discount_rate=discount_rate,
            q_x_seed=q_x_seed,
            q_y_seed=q_y_seed,
            gamma_optimistic=gamma_optimistic,
            gamma_cautious=gamma_cautious,
            lambda_cautious=lambda_cautious,
            s_x_seed=s_x_seed,
            s_y_seed=s_y_seed,
            q_gp_params=q_gp_params,
            s_gp_params=s_gp_params,
            keep_seed_in_data=keep_seed_in_data
        )
        self.violated_constraint = False
        self.updated_safety = False

    def get_next_action(self):
        is_cautious, proba_slice, covar_slice = self.safety_model.level_set(
            self.state,
            lambda_threshold=self.lambda_cautious,
            gamma_threshold=self.gamma_cautious,
            return_proba=True,
            return_covar=True
        )
        is_cautious = is_cautious.squeeze()
        proba_slice = proba_slice.squeeze()

        q_values = self.Q_model[self.state, :]
        action = self.constrained_value_policy.get_action(
            q_values, is_cautious
        )

        self.violated_constraint = False
        if action is None:
            logger.info('No viable action, taking the safest...')
            action = self.safety_maximization_policy.get_action(
                proba_slice
            )
            self.violated_constraint = True

        return action

    def step(self):
        old_state = self.state
        action = self.get_next_action()
        new_state, reward, failed = self.env.step(action)
        if self.training_mode:
            self.Q_model.update(old_state, action, new_state, reward,
                                failed)
            self.safety_model.update(old_state, action, new_state, reward,
                                     failed)
        else:
            # Normally, this if-else clause is also used to update
            # self.updated_safety. Here, we always update safety: hence, this
            # boolean is useless, and we set it to False so the samples are
            # plotted in red
            pass
        self.state = new_state
        self.last_action = action
        return new_state, reward, failed, self.env.done