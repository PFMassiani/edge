import numpy as np
from gpytorch.means import ZeroMean

from edge.agent import Agent
from edge.model.safety_models import MaternSafety
from edge.model.inference.tensorwrap import ensure_tensor
from edge.model.policy_models import DLQRPolicy


def affine_interpolation(t, start, end):
    return start + (end - start) * t


class DLQRController(Agent):
    # TODO allow model perturbation
    def __init__(self, env, perturbations=None):
        perturbations = perturbations if perturbations is not None else {}
        A, B = env.linearization(discrete_time=True, **perturbations)
        n, p = B.shape
        Q = np.eye(n, dtype=A.dtype)
        R = np.eye(p, dtype=B.dtype)

        self.dlqr_policy = DLQRPolicy(env.stateaction_space, A, B, Q, R)

        super().__init__(env)

    def get_next_action(self, *args, **kwargs):
        return self.dlqr_policy(self.state)

    def update_models(self, state, action, new_state, reward, failed, done):
        pass


class OfflineSafetyLearner(Agent):
    def __init__(self, env, s_gp_params, gamma_measure):
        self.gamma_measure_s, self.gamma_measure_e = gamma_measure
        x_seed = s_gp_params.pop('train_x')
        y_seed = s_gp_params.pop('train_y')
        self.safety_model = MaternSafety(
            env,
            gamma_measure=self.gamma_measure_s,
            x_seed=x_seed,
            y_seed=y_seed,
            gp_params=s_gp_params
        )
        self._train_gp_mean_constant = self.gp_mean_constant
        super().__init__(env, self.safety_model)

    @property
    def gp_mean_constant(self):
        if isinstance(self.safety_model.gp.mean_module, ZeroMean):
            return 0
        else:
            return float(
                self.safety_model.gp.mean_module.constant.detach().data[0]
            )

    @gp_mean_constant.setter
    def gp_mean_constant(self, new_constant):
        if isinstance(self.safety_model.gp.mean_module, ZeroMean):
            pass
        else:
            mean_module = self.safety_model.gp.mean_module
            mean_module.initialize(constant=ensure_tensor([new_constant]))
            mean_module.requires_grad = False

    @property
    def gamma_measure(self):
        return self.safety_model.gamma_measure

    @gamma_measure.setter
    def gamma_measure(self, new_gamma_measure):
        self.safety_model.gamma_measure = new_gamma_measure

    @Agent.training_mode.setter
    def training_mode(self, new_training_mode):
        if new_training_mode:
            self.gp_mean_constant = self._train_gp_mean_constant
        else:
            self.gp_mean_constant = 0
        super().training_mode(self, new_training_mode)

    def update_safety_params(self, t):
        self.gamma_measure = affine_interpolation(t, self.gamma_measure_s,
                                                   self.gamma_measure_e)

    def batch_update_models(self, states, actions, new_states, rewards, faileds,
                           dones, measures=None):
        # We redefine everything here because safety_model.update does not
        # handle batch updates
        if measures is None:
            measures = self.safety_model.measure(
                state=new_states,
                lambda_threshold=0,
                gamma_threshold=self.gamma_measure
            )
            measures[faileds] = 0
        stateactions = np.hstack((states, actions))
        self.safety_model.gp.append_data(stateactions, measures)

    def update_models(self, state, action, new_state, reward, failed, done):
        pass

    def get_next_action(self):
        pass
