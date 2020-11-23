from edge.agent import Agent, SafetyLearner
from edge.model.policy_models import SafetyInformationMaximization, \
    RandomPolicy, ConstantPolicy, AffinePolicy, SafetyActiveSampling, \
    SafetyMaximization, SafeProjectionPolicy
from edge.model.safety_models import MaternSafety
from edge.utils import affine_interpolation


class ControlledSafetyLearner(Agent):
    def __init__(self, env, s_gp_params, gamma_cautious, lambda_cautious,
                 gamma_optimistic, checks_safety=True, learn_safety=True,
                 is_free_from_safety=False, always_update_safety=False,
                 safety_model=None,
                 *models):
        self.gamma_cautious_s, self.gamma_cautious_e = gamma_cautious
        self.lambda_cautious_s, self.lambda_cautious_e = lambda_cautious
        self.gamma_optimistic_s, self.gamma_optimistic_e = gamma_optimistic
        self.gamma_cautious = self.gamma_cautious_s
        self.lambda_cautious = self.lambda_cautious_s

        if safety_model is not None:
            self.safety_model = safety_model
        else:
            x_seed = s_gp_params.pop('train_x')
            y_seed = s_gp_params.pop('train_y')
            self.safety_model = MaternSafety(
                env,
                gamma_measure=self.gamma_optimistic_s,
                x_seed=x_seed,
                y_seed=y_seed,
                gp_params=s_gp_params
            )

        super().__init__(env, self.safety_model, *models)
        self.safety_learning_policy = SafetyInformationMaximization(
            env.stateaction_space
        )
        self.safe_projection_policy = SafeProjectionPolicy(
            env.stateaction_space
        )
        self.safety_maximization_policy = SafetyMaximization(
            self.env.stateaction_space
        )
        self.active_sampling_policy = SafetyActiveSampling(
            self.env.stateaction_space
        )
        self.last_controller_action = None
        self.safety_update = None
        self.checks_safety = checks_safety
        self.followed_controller = None
        self.always_update_safety = always_update_safety
        self.violated_constraint = None
        self.is_free_from_safety = is_free_from_safety
        self.learn_safety = learn_safety

    def get_controller_action(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def gamma_optimistic(self):
        return self.safety_model.gamma_measure

    @gamma_optimistic.setter
    def gamma_optimistic(self, new_gamma_optimistic):
        self.safety_model.gamma_measure = new_gamma_optimistic

    @property
    def do_safety_update(self):
        return self.learn_safety and ( True
                # self.always_update_safety
                # or self.violated_constraint
                # or (not self.followed_controller)
                # or self.failed
        )

    def update_safety_params(self, t):
        self.gamma_cautious = affine_interpolation(t, self.gamma_cautious_s,
                                                   self.gamma_cautious_e)
        self.lambda_cautious = affine_interpolation(t, self.lambda_cautious_s,
                                                    self.lambda_cautious_e)
        self.gamma_optimistic = affine_interpolation(t, self.gamma_optimistic_s,
                                                     self.gamma_optimistic_e)

    def __get_projection_with_thresholds(self, lambda_t, gamma_t,
                                         original_action):
        constraints = self.safety_model.level_set(
            self.state,
            lambda_threshold=lambda_t,
            gamma_threshold=gamma_t
        )
        projected_action = self.safe_projection_policy.get_action(
            to_project=original_action,
            constraints=constraints
        )
        return projected_action

    def __get_alternative_with_thresholds(self, lambda_t, gamma_t,
                                          maximize_safety_proba=False,
                                          use_covar_slice=False):
        alt_set, safety_proba, covar_slice, covar_matrix = \
            self.safety_model.level_set(
                self.state,
                lambda_threshold=lambda_t,
                gamma_threshold=gamma_t,
                return_proba=True,
                return_covar=True,
                return_covar_matrix=True,
            )
        if not maximize_safety_proba:
            alt_set = alt_set.squeeze()
            if alt_set.any():
                ctrlr_idx = self.env.action_space.get_index_of(
                    self.last_controller_action, around_ok=True
                )
                if use_covar_slice:
                    alternative = self.active_sampling_policy(
                        covar_slice.squeeze(), alt_set
                    )
                else:
                    alternative = self.safety_learning_policy.get_action(
                        covar_matrix[ctrlr_idx, :].squeeze(), alt_set
                    )
                return alternative
            else:
                return None
        else:
            safety_proba = safety_proba.squeeze()
            return self.safety_maximization_policy.get_action(safety_proba)

    def get_next_action(self):
        self.followed_controller = True
        self.violated_constraint = False
        self.last_controller_action = self.get_controller_action()
        action = self.last_controller_action
        if self.checks_safety:
            controller_is_cautious = self.safety_model.is_in_level_set(
                self.state, action, self.lambda_cautious, self.gamma_cautious
            )
            if not controller_is_cautious:
                if self.is_free_from_safety:
                    self.violated_constraint = True
                else:
                    # alternative = self.__get_alternative_with_thresholds(
                    #     self.lambda_cautious, self.gamma_cautious,
                    #     use_covar_slice=False
                    # )
                    alternative = self.__get_projection_with_thresholds(
                        self.lambda_cautious, self.gamma_cautious, action
                    )
                    if alternative is not None:
                        # We found a cautious alternative
                        self.violated_constraint = False
                        self.followed_controller = False
                        action = alternative
                    else:
                        self.violated_constraint = True
                        self.followed_controller = False
                        # alternative = self.__get_alternative_with_thresholds(
                        #     0., self.gamma_optimistic
                        # )
                        alternative = self.__get_projection_with_thresholds(
                            0., self.gamma_optimistic, action
                        )
                        if alternative is not None:
                            # We found an optimistic alternative
                            action = alternative
                        else:
                            # No cautious or optimistic action available:
                            # maximize safety probability
                            action = self.__get_alternative_with_thresholds(
                                0., self.gamma_optimistic,
                                maximize_safety_proba=True
                            )
        return action

    def update_models(self, state, action, next_state, reward, failed, done):
        return self.safety_model.update(state, action, next_state, reward,
                                        failed, done)

    def step(self):
        """
        Chooses an action according to the policy, takes a step in the Environment, and updates the models. The action
        taken is available in self.last_action.
        :return: new_state, reward, failed
        """
        old_state = self.state
        self.last_action = self.get_next_action()
        self.state, reward, failed = self.env.step(self.last_action)
        done = self.env.done
        if self.training_mode and self.do_safety_update:
            self.safety_update = self.update_models(
                old_state, self.last_action, self.state, reward, failed,
                done
            )
        else:
            self.safety_update = None
        return self.state, reward, failed, done


class RandomSafetyLearner(ControlledSafetyLearner):
    def __init__(self, env, s_gp_params, gamma_cautious, lambda_cautious,
                 gamma_optimistic, safety_options=None):
        if safety_options is None:
            safety_options = {}
        super().__init__(env, s_gp_params, gamma_cautious, lambda_cautious,
                         gamma_optimistic, **safety_options)
        self.policy = RandomPolicy(env.stateaction_space)

    def get_controller_action(self, *args, **kwargs):
        return self.policy(self.state)

    @staticmethod
    def load(env, mpath, gamma_cautious, lambda_cautious, **safety_options):
        safety_model = MaternSafety.load(
            mpath, env, gamma_measure=None, x_seed=None, y_seed=None
        )
        safety_options["safety_model"] = safety_model
        gamma_optimistic = (safety_model.gamma_measure,
                            safety_model.gamma_measure)
        agent = RandomSafetyLearner(env, {}, gamma_cautious, lambda_cautious,
                                    gamma_optimistic, safety_options)
        return agent


class ConstantSafetyLearner(ControlledSafetyLearner):
    def __init__(self, env, constant_action, s_gp_params, gamma_cautious,
                 lambda_cautious, gamma_optimistic, safety_options=None):
        if safety_options is None:
            safety_options = {}
        super().__init__(env, s_gp_params, gamma_cautious, lambda_cautious,
                         gamma_optimistic, **safety_options)
        self.policy = ConstantPolicy(env.stateaction_space, constant_action)

    def get_controller_action(self, *args, **kwargs):
        return self.policy(self.state)


class AffineSafetyLearner(ControlledSafetyLearner):
    def __init__(self, env, offset, jacobian, s_gp_params, gamma_cautious,
                 lambda_cautious, gamma_optimistic, safety_options=None):
        if safety_options is None:
            safety_options = {}
        super().__init__(env, s_gp_params, gamma_cautious, lambda_cautious,
                         gamma_optimistic, **safety_options)
        self.policy = AffinePolicy(env.stateaction_space, offset, jacobian)

    def get_controller_action(self, *args, **kwargs):
        return self.policy(self.state)

    @staticmethod
    def load(env, mpath, offset, jacobian, gamma_cautious,
             lambda_cautious, **safety_options):
        safety_model = MaternSafety.load(
            mpath, env, gamma_measure=None, x_seed=None, y_seed=None
        )
        safety_options["safety_model"] = safety_model
        gamma_optimistic = (safety_model.gamma_measure,
                            safety_model.gamma_measure)
        agent = AffineSafetyLearner(env, offset, jacobian, {}, gamma_cautious,
                                    lambda_cautious, gamma_optimistic,
                                    safety_options)
        return agent


class CoRLSafetyLearner(ControlledSafetyLearner):
    def __init__(self, env, s_gp_params, gamma_cautious, lambda_cautious,
                 gamma_optimistic, base_controller):
        super().__init__(env, s_gp_params, gamma_cautious, lambda_cautious,
                         gamma_optimistic, checks_safety=True,
                         learn_safety=True, is_free_from_safety=False,
                         always_update_safety=True, safety_model=None)
        self.policy = base_controller
        self.active_sampling_policy = SafetyActiveSampling(
            self.env.stateaction_space
        )
        self.safety_maximization_policy = SafetyMaximization(
            self.env.stateaction_space
        )

    def get_controller_action(self, *args, **kwargs):
        return self.policy(self.state)

    def __get_active_sampling_with_thresholds(self, lambda_t, gamma_t,
                                              maximize_safety_proba=False):
        in_set, proba_slice, covar_slice = self.safety_model.level_set(
            self.state,
            lambda_threshold=lambda_t,
            gamma_threshold=gamma_t,
            return_proba=True,
            return_covar=True
        )
        if not maximize_safety_proba:
            in_set = in_set.squeeze()
            covar_slice = covar_slice.squeeze()

            action = self.active_sampling_policy.get_action(
                covar_slice, in_set
            )
            return action
        else:
            proba_slice = proba_slice.squeeze()
            return self.safety_maximization_policy.get_action(proba_slice)

    def get_next_action(self):
        if self.training_mode:
            self.last_controller_action = self.get_controller_action()
            self.followed_controller = False
            self.violated_constraint = False
            action = self.__get_active_sampling_with_thresholds(
                self.lambda_cautious, self.gamma_cautious
            )
            if action is not None:
                self.violated_constraint = False
            else:
                self.violated_constraint = True
                action = self.__get_active_sampling_with_thresholds(
                    0., self.gamma_optimistic
                )
            if action is None:
                action = self.__get_active_sampling_with_thresholds(
                    0., self.gamma_optimistic, maximize_safety_proba=True
                )
            return action
        else:
            return super().get_next_action()
