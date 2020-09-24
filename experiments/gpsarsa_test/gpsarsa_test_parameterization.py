import numpy as np

from edge.agent import Agent
from edge.model.value_models import MaternGPSARSA
from edge.model.safety_models import MaternSafety
from edge.model.policy_models.bayesian import \
    ExpectedImprovementPolicy, SafetyInformationMaximization
from edge.model.policy_models import SafetyMaximization

from edge.envs import Hovership
from edge.reward import AffineReward, ConstantReward


class SARSALearner(Agent):
    """
    Defines an Agent modelling the Q-Values with a MaternGPSARSA and following
    the maximal expected improvement policy.
    """
    def __init__(self, env, xi, keep_seed_in_data, q_gp_params,
                 s_gp_params=None, gamma_cautious=None, lambda_cautious=None,
                 gamma_optimistic=None):
        """
        Initializer
        :param env: the environment
        :param xi: the xi parameter of the expected improvement policy
        :param keep_seed_in_data: whether to keep the seed in the dataset.
            Should be True, otherwise GPyTorch fails.
        :param gp_params: params passed to the MaternGP constructor. Parameter
            `value_structure_discount_factor` is mandatory.
        """
        Q_model = MaternGPSARSA(env, **q_gp_params)
        self.has_safety_model = s_gp_params is not None
        if self.has_safety_model:
            x_seed = s_gp_params.pop('train_x')
            y_seed = s_gp_params.pop('train_y')
            safety_model = MaternSafety(
                env,
                gamma_measure=gamma_optimistic,
                x_seed=x_seed,
                y_seed=y_seed,
                gp_params=s_gp_params
            )
            super(SARSALearner, self).__init__(env, Q_model, safety_model)
            safety_learning_policy = SafetyInformationMaximization(
                env.stateaction_space
            )
            safety_maximization_policy = SafetyMaximization(
                env.stateaction_space
            )
        else:
            super(SARSALearner, self).__init__(env, Q_model)
            safety_model = None
            safety_learning_policy = None
            safety_maximization_policy = None

        self.Q_model = Q_model
        self.safety_model = safety_model
        self.gamma_cautious = gamma_cautious
        self.lambda_cautious = lambda_cautious
        self.policy = ExpectedImprovementPolicy(env.stateaction_space, xi)
        self.safety_learning_policy = safety_learning_policy
        self.safety_maximization_policy = safety_maximization_policy
        self.keep_seed_in_data = keep_seed_in_data
        if not keep_seed_in_data:
            self.Q_model.empty_data()
            if self.has_safety_model:
                self.safety_model.empty_data()
        self.update_safety_model = False

    def _compute_best_sample(self):
        predictions = self.Q_model[self.state, :]
        return np.max(predictions)

    @property
    def xi(self):
        """
        Returns the xi parameter of the policy
        :return: epsilon parameter
        """
        return self.policy.xi

    @xi.setter
    def xi(self, new_xi):
        """
        Sets the epsilon parameter of the policy
        """
        self.policy.xi = new_xi

    @property
    def gamma_optimistic(self):
        if self.has_safety_model:
            return self.safety_model.gamma_measure
        else:
            return None

    @gamma_optimistic.setter
    def gamma_optimistic(self, new_gamma_optimistic):
        if self.has_safety_model:
            self.safety_model.gamma_measure = new_gamma_optimistic
        else:
            pass

    def get_next_action(self):
        """
        If the Agent has no safety model, this returns the action `a` with the
        maximal expected improvement, `a_best`.
        If the Agent has a safety model:
        - If no cautious action is available, `a` is the action that maximizes
        the safety probability
        - Otherwise:
            - If the action `a_best` is cautious, `a = a_best`
            - Otherwise, `a` is the cautious action with the maximal posterior
            covariance with `a_best`
        :return: the chosen action
        """
        self.update_safety_model = False
        q_values, covar = self.Q_model.query(
            index=tuple(self.env.stateaction_space.get_stateaction(
                self.state, slice(None, None, None)
            )),
            return_covar=True
        )

        constraints = None
        if self.has_safety_model:
            constraints_list, proba_list, covar_matrix = self.safety_model.\
                level_set(
                    self.state,
                    lambda_threshold=[self.lambda_cautious, 0],
                    gamma_threshold=[self.gamma_cautious, self.gamma_optimistic],
                    return_proba=True,
                    return_covar=False,
                    return_covar_matrix=True,
            )
            is_cautious, is_optimistic = constraints_list
            is_cautious = is_cautious.squeeze()
            is_optimistic = is_optimistic.squeeze()
            proba_optimistic = proba_list[0].squeeze()
            constraints = is_optimistic

        action = self.policy.get_action(
            mean=q_values,
            covar=covar,
            best_sample=self._compute_best_sample(),
            constraints=constraints,
        )

        if self.has_safety_model and action is not None:
            action_idx = self.env.action_space.get_index_of(action)
            if not is_cautious[action_idx]:
                action = self.safety_learning_policy.get_action(
                    covar_matrix[action_idx, :], is_cautious
                )
                self.update_safety_model = True
        if self.has_safety_model and action is None:
            action = self.safety_maximization_policy.get_action(
                proba_optimistic
            )
            self.update_safety_model = True

        return action

    def update_models(self, state, action, next_state, reward, failed):
        self.Q_model.update(state, action, next_state, reward, failed)
        if self.update_safety_model:
            self.safety_model.update(state, action, next_state, reward, failed)

    def fit_models(self, train_x, train_y, epochs, **optimizer_kwargs):
        self.Q_model.fit(train_x, train_y, epochs, **optimizer_kwargs)
        if not self.keep_seed_in_data:
            self.Q_model.empty_data()

    def step(self):
        new_state, reward, failed = super(SARSALearner, self).step()
        return new_state, reward, failed, self.env.done


class LowGoalHovership(Hovership):
    def __init__(self, dynamics_parameters=None, steps_done_threshold=None):
        super(LowGoalHovership, self).__init__(
            random_start=True,
            dynamics_parameters=dynamics_parameters,
            steps_done_threshold=steps_done_threshold,
        )

        reward = AffineReward(self.stateaction_space, [(10, 0), (0, 0)])
        self.reward = reward


class PenalizedHovership(LowGoalHovership):
    def __init__(self, penalty_level=100, dynamics_parameters=None,
                 steps_done_threshold=None):
        super(PenalizedHovership, self).__init__(dynamics_parameters,
                                                 steps_done_threshold)

        def penalty_condition(state, action, new_state, reward):
            return self.is_failure_state(new_state)

        penalty = ConstantReward(self.reward.stateaction_space, -penalty_level,
                                 reward_condition=penalty_condition)

        self.reward += penalty