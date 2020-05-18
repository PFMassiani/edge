import numpy as np

from edge.model.safety_models import MaternSafety
from . import Agent


class SafetyLearner(Agent):
    def __init__(self, env, gamma_optimistic, gamma_cautious, lambda_cautious,
                 x_seed, y_seed, gp_params=None, keep_seed_in_data=False):
        safety_model = MaternSafety(env, gamma_optimistic,
                                    x_seed, y_seed, gp_params)
        super(SafetyLearner, self).__init__(env, safety_model)

        self.safety_model = safety_model

        self.gamma_cautious = gamma_cautious
        self.lambda_cautious = lambda_cautious

        self.keep_seed_in_data = keep_seed_in_data

    @property
    def gamma_optimistic(self):
        return self.safety_model.gamma_measure

    def get_next_action(self):
        is_cautious, proba_slice, covar_slice = self.safety_model.level_set(
            self.state,
            lambda_threshold=self.lambda_cautious,
            gamma_threshold=self.gamma_cautious,
            return_proba=True,
            return_covar=True
        )

        if not is_cautious.any():
            action_idx = np.unravel_index(
                np.argmax(proba_slice),
                shape=self.env.action_space.shape
            )
            action = self.env.action_space[action_idx]

        else:
            cautious_indexes = np.argwhere(is_cautious)
            most_variance_action = np.argmax(
                covar_slice[cautious_indexes]
            )
            action_idx = tuple(cautious_indexes[most_variance_action])
            action = self.env.action_space[action_idx]

        return action

    def update_models(self, state, action, next_state, reward, failed):
        self.safety_model.update(state, action, next_state, reward, failed)

    def fit_models(self, train_x, train_y, epochs, **optimizer_kwargs):
        self.safety_model.fit(train_x, train_y, epochs, **optimizer_kwargs)
        if not self.keep_seed_in_data:
            self.safety_model.empty_data()
