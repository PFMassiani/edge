import numpy as np

from edge.model.value_models import GPQLearning
from . import Agent


class QLearner(Agent):
    def __init__(self, env, greed, step_size, discount_rate, x_seed, y_seed,
                 gp_params=None, keep_seed_in_data=True):
        Q_model = GPQLearning(env, step_size, discount_rate,
                              x_seed=x_seed, y_seed=y_seed,
                              gp_params=gp_params)
        super(QLearner, self).__init__(env, Q_model)

        self.Q_model = Q_model
        self.__greed = greed
        self.keep_seed_in_data = keep_seed_in_data
        if not keep_seed_in_data:
            self.Q_model.empty_data()

    @property
    def greed(self):
        return self.__greed

    @greed.setter
    def greed(self, new_greed):
        self.__greed = np.clip(new_greed, 0, 1)

    def get_next_action(self):
        q_values = self.Q_model[self.state, :].reshape(
            self.env.action_space.shape
        )  # nA1 x nA2 x ... x nAp

        nA = np.prod(q_values.shape)
        probabilities = np.ones_like(q_values) * self.greed / nA

        best_value_action = np.unravel_index(
            np.argmax(q_values), shape=q_values.shape
        )
        probabilities[best_value_action] += 1 - self.greed

        action_index = np.unravel_index(
            np.random.choice(nA, p=probabilities.ravel()),
            probabilities.shape
        )
        action = self.env.action_space[action_index]

        return action

    def update_models(self, state, action, next_state, reward, failed):
        self.Q_model.update(state, action, next_state, reward, failed)

    def fit_models(self, train_x, train_y, epochs, **optimizer_kwargs):
        self.Q_model.fit(train_x, train_y, epochs, **optimizer_kwargs)
        if not self.keep_seed_in_data:
            self.Q_model.empty_data()
