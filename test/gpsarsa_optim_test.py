import unittest
import gpytorch
import numpy as np
from edge.model.value_models import MaternGPSARSA
from edge.reward import ConstantReward
from edge.envs import DiscreteHovership


class UnviableHovership(DiscreteHovership):
    def __init__(self, max_altitude):
        dynamics_parameters = {
            'ground_gravity': 2,
            'gravity_gradient': 0,
            'max_thrust': 1,
            'max_altitude': max_altitude,
            'minimum_gravity_altitude': 0,
            'maximum_gravity_altitude': 5
        }
        super(UnviableHovership, self).__init__(
            random_start=True,
            dynamics_parameters=dynamics_parameters,
            reward_done_threshold=None,
            steps_done_threshold=None,
        )

        def condition(state, action, new_state, reward):
            return state.squeeze() == 1

        zero_reward = ConstantReward(self.stateaction_space, 0)
        final_reward = ConstantReward(self.stateaction_space, 1.,
                                      reward_condition=condition)
        self.reward = zero_reward + final_reward


class TestGPSARSAOptim(unittest.TestCase):
    def value_cv_routine(self, lazily_evaluate):
        max_altitude = 5
        gamma = 1
        env = UnviableHovership(max_altitude=max_altitude)
        gp_params = {
            'train_x': np.array([[5., 1.], [4., 1.]]),
            'train_y': np.array([0., 0.]),
            'outputscale_prior': (1, 1),
            'lengthscale_prior': (0.1, 2),
            'noise_prior': (1e-3, 1),
            'dataset_type': None,
            'dataset_params': None,
            'value_structure_discount_factor': gamma,
        }
        q_model = MaternGPSARSA(env, **gp_params)
        action = env.action_space[1]  # full thrusters
        for t in range(10):
            while env.done:
                env.reset()
            s = env.s
            s_, r, f = env.step(action)
            q_model.update(s, action, s_, r, f, env.done)
        with gpytorch.settings.lazily_evaluate_kernels(lazily_evaluate):
            q_model.fit(epochs=10, lr=1)
            q_model.fit(epochs=10, lr=0.1)
            # q_model.fit(epochs=10, lr=0.01)
        q_values = q_model[:, :].reshape(env.stateaction_space.shape)
        values = q_values.max(axis=1)[1:]
        true_values = np.array([1. for t in range(1, max_altitude + 1)])

        l = q_model.gp.covar_module.base_kernel.base_kernel.lengthscale
        o = q_model.gp.covar_module.base_kernel.outputscale
        n = q_model.gp.likelihood.noise

        params = f'Lengthscale: {l}\nNoise: {n}'
        error_message = (f'Values are not close:\nTrue: {true_values}\n'
                         f'Computed: {values}\n' + params)
        self.assertLess(np.abs(values - true_values).max(), 1e-2, error_message)
        self.assertLess(1, l[0][0], f'Lengthscale is too small: {l[0][0]}')

    def test_value_cv_lazy_evaluation(self):
        self.value_cv_routine(True)

    def test_value_cv_active_evaluation(self):
        self.value_cv_routine(False)