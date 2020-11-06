import numpy as np

from edge.agent import Agent
from edge.model.policy_models import DLQRPolicy, SafetyInformationMaximization
from edge.utils import affine_interpolation

from learned_mean_model import LearnedMeanMaternSafety


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
            self.safety_model = LearnedMeanMaternSafety(
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
        return self.learn_safety and (
                self.always_update_safety or
                self.violated_constraint or
                (not self.followed_controller) or
                self.failed
        )

    def update_safety_params(self, t):
        self.gamma_cautious = affine_interpolation(t, self.gamma_cautious_s,
                                                   self.gamma_cautious_e)
        self.lambda_cautious = affine_interpolation(t, self.lambda_cautious_s,
                                                    self.lambda_cautious_e)
        self.gamma_optimistic = affine_interpolation(t, self.gamma_optimistic_s,
                                                     self.gamma_optimistic_e)

    def get_next_action(self):
        self.followed_controller = True
        self.violated_constraint = False
        action = self.get_controller_action()
        if self.checks_safety:
            controller_is_cautious = self.safety_model.is_in_level_set(
                self.state, action, self.lambda_cautious, self.gamma_cautious
            )
            if not controller_is_cautious:
                if self.is_free_from_safety:
                    self.violated_constraint = True
                else:
                    cautious_set, covar_matrix = self.safety_model.level_set(
                        self.state,
                        lambda_threshold=self.lambda_cautious,
                        gamma_threshold=self.gamma_cautious,
                        return_proba=False,
                        return_covar=False,
                        return_covar_matrix=True,
                    )
                    cautious_set = cautious_set.squeeze()
                    if cautious_set.any():
                        action_idx = self.env.action_space.get_index_of(
                            action, around_ok=True
                        )
                        action = self.safety_learning_policy.get_action(
                            covar_matrix[action_idx, :].squeeze(), cautious_set
                        )
                        self.followed_controller = False
                    else:
                        # Keep following the initial controller if no action is safe
                        self.violated_constraint = True
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


class DLQRSafetyLearner(ControlledSafetyLearner):
    def __init__(self, env, s_gp_params, gamma_cautious, lambda_cautious,
                 gamma_optimistic, perturbations=None, safety_model=None,
                 **kwargs):
        perturbations = perturbations if perturbations is not None else {}
        A, B = env.linearization(discrete_time=True, **perturbations)
        n, p = B.shape
        Q = np.eye(n, dtype=A.dtype)
        R = np.eye(p, dtype=B.dtype)

        self.dlqr_policy = DLQRPolicy(env.stateaction_space, A, B, Q, R)

        super().__init__(env, s_gp_params, gamma_cautious, lambda_cautious,
                         gamma_optimistic, safety_model=safety_model, **kwargs)

    def get_controller_action(self, *args, **kwargs):
        return self.dlqr_policy(self.state)

    @staticmethod
    def load(load_path, env, x_seed, y_seed, gamma_cautious, lambda_cautious,
             gamma_optimistic, perturbations=None, **kwargs):
        safety_model = LearnedMeanMaternSafety.load(
            load_path, env, gamma_optimistic, x_seed, y_seed
        )
        if gamma_optimistic is None:
            gamma_optimistic = safety_model.gamma_measure

        def get_true_arg(arg):
            return (arg, arg) if isinstance(arg, float) else arg

        return DLQRSafetyLearner(
            env, {},
            get_true_arg(gamma_cautious),
            get_true_arg(lambda_cautious),
            get_true_arg(gamma_optimistic),
            perturbations, safety_model=safety_model,
            **kwargs
        )


class PickyDLQRSafetyLearner(DLQRSafetyLearner):
    def __init__(self, env, s_gp_params, gamma_cautious, lambda_cautious,
                 gamma_optimistic, perturbations=None, safety_model=None,
                 pick_radius=0.01,
                 **kwargs):
        super().__init__(env, s_gp_params, gamma_cautious, lambda_cautious,
                         gamma_optimistic, perturbations, safety_model,
                         **kwargs)
        self.pick_radius = pick_radius
        self._last_added_sa = None

    def update_models(self, state, action, next_state, reward, failed, done):
        sa = np.hstack((state.reshape(-1), action.reshape(-1)))
        if self._last_added_sa is None or failed or done or (
            np.linalg.norm(sa - self._last_added_sa) >= self.pick_radius
        ):
            self._last_added_sa = sa
            super().update_models(state, action, next_state, reward, failed,
                                  done)

    @staticmethod
    def load(load_path, env, x_seed, y_seed, gamma_cautious, lambda_cautious,
             gamma_optimistic, perturbations=None, **kwargs):
        safety_model = LearnedMeanMaternSafety.load(
            load_path, env, gamma_optimistic, x_seed, y_seed
        )
        if gamma_optimistic is None:
            gamma_optimistic = safety_model.gamma_measure

        def get_true_arg(arg):
            return (arg, arg) if isinstance(arg, float) else arg

        return PickyDLQRSafetyLearner(
            env, {},
            get_true_arg(gamma_cautious),
            get_true_arg(lambda_cautious),
            get_true_arg(gamma_optimistic),
            perturbations, safety_model=safety_model,
            **kwargs
        )