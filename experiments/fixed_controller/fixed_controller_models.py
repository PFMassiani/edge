import numpy as np
from edge.model.safety_models import MaternSafety


class ZeroUpdateMaternSafety(MaternSafety):
    def __init__(self, env, gamma_measure, x_seed, y_seed, gp_params=None):
        """
        Initializer
        :param env: the environment
        :param gamma_measure: the gamma coefficient used by the measure. It corresponds to gamma_optimistic
        :param x_seed: the seed input of the GP: a list of stateactions
        :param y_seed: the seed output of the GP: a list of floats
        :param gp_params: the parameters defining the GP. See edge.models.inference.MaternGP for more information
        """
        if gp_params is None:
            gp_params = {}
        if gp_params.get('mean_constant') is None:
            gp_params['mean_constant'] = 1.
        super().__init__(env, gamma_measure, x_seed, y_seed, gp_params)

    def update(self, state, action, new_state, reward, failed, done):
        """
        Updates the underlying GP with the measure computation update
        :param state: the previous state
        :param action: the action taken
        :param new_state: the new state
        :param reward: the reward incurred
        :param failed: whether the agent has failed
        """
        if not failed:
            update_value = self.measure(
                state=new_state,
                lambda_threshold=0,
                gamma_threshold=self.gamma_measure
            )
        else:
            update_value = np.array([0.])
        is_zero = update_value[0] == 0
        if is_zero:
            stateaction = self.env.stateaction_space[state, action]
            self.gp.append_data(stateaction, update_value, forgettable=[False],
                                make_forget=[False],
                                unskippable=[failed])
        return is_zero, update_value


class TDMaternSafety(MaternSafety):
    def __init__(self, env, gamma_measure, x_seed, y_seed, gp_params=None):
        """
        Initializer
        :param env: the environment
        :param gamma_measure: the gamma coefficient used by the measure. It corresponds to gamma_optimistic
        :param x_seed: the seed input of the GP: a list of stateactions
        :param y_seed: the seed output of the GP: a list of floats
        :param gp_params: the parameters defining the GP. See edge.models.inference.MaternGP for more information
        """
        if gp_params is None:
            gp_params = {}
        if gp_params.get('value_structure_discount_factor') is None:
            gp_params['value_structure_discount_factor'] = 1
        if gp_params.get('mean_constant') is None:
            gp_params['mean_constant'] = 1.
        super().__init__(env, gamma_measure, x_seed, y_seed, gp_params)

    def update(self, state, action, new_state, reward, failed, done):
        """
        Updates the underlying GP with the measure computation update
        :param state: the previous state
        :param action: the action taken
        :param new_state: the new state
        :param reward: the reward incurred
        :param failed: whether the agent has failed
        """
        update_value = np.array([0.])
        stateaction = self.env.stateaction_space[state, action]
        self.gp.append_data(stateaction, update_value, is_terminal=[done])
        return update_value
