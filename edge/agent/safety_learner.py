import numpy as np

from edge.model.safety_models import MaternSafety
from . import Agent
from .policy import SafetyActiveSampling, SafetyMaximization


class SafetyLearner(Agent):
    def __init__(self, env, gamma_optimistic, gamma_cautious, lambda_cautious,
                 x_seed, y_seed, gp_params=None, keep_seed_in_data=True):
        safety_model = MaternSafety(env, gamma_optimistic,
                                    x_seed, y_seed, gp_params)
        super(SafetyLearner, self).__init__(env, safety_model)

        self.safety_model = safety_model

        self.active_sampling_policy = SafetyActiveSampling(env)
        self.safety_maximization_policy = SafetyMaximization(env)

        self.gamma_cautious = gamma_cautious
        self.lambda_cautious = lambda_cautious

        self.keep_seed_in_data = keep_seed_in_data

    @property
    def gamma_optimistic(self):
        return self.safety_model.gamma_measure

    def get_random_safe_state(self):
        measure = self.safety_model.measure(
            state=None,
            lambda_threshold=self.lambda_cautious,
            gamma_threshold=self.gamma_cautious
        ).squeeze()
        safe_states = measure > 0
        if not safe_states.any():
            return None
        else:
            safe_indexes = np.argwhere(safe_states).squeeze()
            chosen_index = safe_indexes[np.random.choice(len(safe_indexes))]
            return self.env.state_space[np.unravel_index(
                chosen_index, self.env.state_space.shape
            )]

    def get_next_action(self):
        is_cautious, proba_slice, covar_slice = self.safety_model.level_set(
            self.state,
            lambda_threshold=self.lambda_cautious,
            gamma_threshold=self.gamma_cautious,
            return_proba=True,
            return_covar=True
        )
        action = self.active_sampling_policy.get_action(
            covar_slice, is_cautious
        )
        if action is None:
            action = self.safety_maximization_policy.get_action(proba_slice)

        return action

    def update_models(self, state, action, next_state, reward, failed):
        self.safety_model.update(state, action, next_state, reward, failed)

    def fit_models(self, train_x, train_y, epochs, **optimizer_kwargs):
        self.safety_model.fit(train_x, train_y, epochs, **optimizer_kwargs)
        if not self.keep_seed_in_data:
            self.safety_model.empty_data()
