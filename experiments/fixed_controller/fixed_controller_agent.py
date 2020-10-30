import numpy as np

from edge.agent import Agent
from edge.model.safety_models import MaternSafety
from edge.model.policy_models import SafetyInformationMaximization, RandomPolicy

# noinspection PyUnresolvedReferences
from dlqr_policy import DLQRPolicy
# noinspection PyUnresolvedReferences
from fixed_controller_models import ZeroUpdateMaternSafety, TDMaternSafety


def affine_interpolation(t, start, end):
    return start + (end - start) * t


class ControlledSafetyLearner(Agent):
    def __init__(self, env, s_gp_params, gamma_cautious, lambda_cautious,
                 gamma_optimistic,
                 *models, model_type='default', always_update_safety=False):
        self.gamma_cautious_s, self.gamma_cautious_e = gamma_cautious
        self.lambda_cautious_s, self.lambda_cautious_e = lambda_cautious
        self.gamma_optimistic_s, self.gamma_optimistic_e = gamma_optimistic
        self.gamma_cautious = self.gamma_cautious_s
        self.lambda_cautious = self.lambda_cautious_s

        x_seed = s_gp_params.pop('train_x')
        y_seed = s_gp_params.pop('train_y')
        if model_type == 'zero':
            SafetyMeasure = ZeroUpdateMaternSafety
        elif model_type == 'td':
            SafetyMeasure = TDMaternSafety
        else:
            SafetyMeasure = MaternSafety
        self.safety_model = SafetyMeasure(
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
        self.followed_controller = None
        self.always_update_safety = always_update_safety
        self.violated_constraint = None

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
        return self.always_update_safety or self.violated_constraint or \
               (not self.followed_controller) or self.failed

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
        controller_is_cautious = self.safety_model.is_in_level_set(
            self.state, action, self.lambda_cautious, self.gamma_cautious
        )
        if not controller_is_cautious:
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
                action_idx = self.env.action_space.get_index_of(action,
                                                                around_ok=True)
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
    # TODO allow model perturbation
    def __init__(self, env, s_gp_params, gamma_cautious, lambda_cautious,
                 gamma_optimistic, linearized_control_frequency=None,
                 perturbations=None, model_type='default'):
        if linearized_control_frequency is None:
            linearized_control_frequency = env.control_frequency
        tmp_control_frequency = env.control_frequency
        env.control_frequency = linearized_control_frequency
        perturbations = perturbations if perturbations is not None else {}
        A, B = env.linearization(discrete_time=True, **perturbations)
        env.control_frequency = tmp_control_frequency
        n, p = B.shape
        Q = np.eye(n, dtype=A.dtype)
        R = np.eye(p, dtype=B.dtype)

        self.dlqr_policy = DLQRPolicy(env.stateaction_space, A, B, Q, R)

        super().__init__(env, s_gp_params, gamma_cautious, lambda_cautious,
                         gamma_optimistic, model_type=model_type)

    def get_controller_action(self, *args, **kwargs):
        return self.dlqr_policy(self.state)


class RandomSafetyLearner(ControlledSafetyLearner):
    def __init__(self, env, s_gp_params, gamma_cautious, lambda_cautious,
                 gamma_optimistic, model_type='default'):
        super().__init__(env, s_gp_params, gamma_cautious, lambda_cautious,
                         gamma_optimistic, model_type=model_type)
        self.policy = RandomPolicy(env.stateaction_space)

    def get_controller_action(self, *args, **kwargs):
        return self.policy(self.state)


class FreeRandomSafetyLearner(RandomSafetyLearner):
    def get_next_action(self):
        self.followed_controller = True
        return self.get_controller_action()