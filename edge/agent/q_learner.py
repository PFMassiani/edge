import numpy as np

from edge.model.value_models import GPQLearning
from . import Agent
from .policy import EpsilonGreedy, ConstrainedEpsilonGreedy, SafetyMaximization
from edge.error import NoActionError


class QLearner(Agent):
    def __init__(self, env, greed, step_size, discount_rate, x_seed, y_seed,
                 gp_params=None, keep_seed_in_data=True):
        Q_model = GPQLearning(env, step_size, discount_rate,
                              x_seed=x_seed, y_seed=y_seed,
                              gp_params=gp_params)
        super(QLearner, self).__init__(env, Q_model)

        self.Q_model = Q_model
        self.policy = EpsilonGreedy(env, greed)
        self.keep_seed_in_data = keep_seed_in_data
        if not keep_seed_in_data:
            self.Q_model.empty_data()

    def get_next_action(self):
        q_values = self.Q_model[self.state, :]
        action = self.policy.get_action(q_values)
        return action

    def update_models(self, state, action, next_state, reward, failed):
        self.Q_model.update(state, action, next_state, reward, failed)

    def fit_models(self, train_x, train_y, epochs, **optimizer_kwargs):
        self.Q_model.fit(train_x, train_y, epochs, **optimizer_kwargs)
        if not self.keep_seed_in_data:
            self.Q_model.empty_data()


class ConstrainedQLearner(Agent):
    def __init__(self, env, safety_measure, greed, step_size, discount_rate,
                 safety_threshold,
                 x_seed, y_seed, gp_params=None, keep_seed_in_data=True):
        Q_model = GPQLearning(env, step_size, discount_rate,
                              x_seed=x_seed, y_seed=y_seed,
                              gp_params=gp_params)
        super(ConstrainedQLearner, self).__init__(env, Q_model)

        self.Q_model = Q_model
        self.safety_measure = safety_measure
        self.constrained_value_policy = ConstrainedEpsilonGreedy(
            self.env.stateaction_space, greed)
        self.safety_maximization_policy = SafetyMaximization(
            self.safety_measure.stateaction_space)
        self.safety_threshold = safety_threshold
        self.keep_seed_in_data = keep_seed_in_data
        if not keep_seed_in_data:
            self.Q_model.empty_data()

    def get_next_action(self):
        all_actions = self.Q_model.env.action_space[:].reshape(-1, 1)
        action_is_viable = [
            self.safety_measure.measure(self.state, a) > self.safety_threshold
            for a in all_actions
        ]
        q_values = self.Q_model[self.state, :]

        action = self.constrained_value_policy.get_action(
            q_values, action_is_viable
        )
        if action is None:
            safety_values = self.safety_measure.measure(
                self.state, slice(None, None, None)
            )
            action = self.safety_maximization_policy.get_action(safety_values)
        if action is None:
            raise NoActionError('The agent could not find a suitable action')

        return action

    def get_random_safe_state(self):
        is_viable = self.safety_measure.state_measure > 0
        viable_indexes = np.argwhere(is_viable).squeeze()
        state_index = viable_indexes[np.random.choice(len(viable_indexes))]
        state = self.safety_measure.stateaction_space.state_space[state_index]
        return state

    def update_models(self, state, action, next_state, reward, failed):
        self.Q_model.update(state, action, next_state, reward, failed)

    def fit_models(self, train_x, train_y, epochs, **optimizer_kwargs):
        self.Q_model.fit(train_x, train_y, epochs, **optimizer_kwargs)
        if not self.keep_seed_in_data:
            self.Q_model.empty_data()
