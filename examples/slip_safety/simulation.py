from pathlib import Path
import numpy as np

from edge import ModelLearningSimulation
from edge.envs import Slip
from edge.agent import SafetyLearner
from edge.reward import AffineReward, ConstantReward  # should not matter
from edge.model.safety_models import SafetyTruth, SafetyMeasure
# from edge.model.value_models import GPQLearning
from edge.graphics.plotter import SafetyPlotter, DetailedSafetyPlotter


class LowGoalSlip(Slip):
    # * This goal incentivizes the agent to run fast
    def __init__(self, dynamics_parameters=None):
        super(LowGoalSlip, self).__init__(
            dynamics_parameters=dynamics_parameters
        )

        reward = AffineReward(self.stateaction_space, [(1, 0), (0, 0)])
        self.reward = reward


class PenalizedSlip(LowGoalSlip):
    def __init__(self, penalty_level=100, dynamics_parameters=None):
        super(PenalizedSlip, self).__init__(dynamics_parameters)

        def penalty_condition(state, action, new_state, reward):
            return self.is_failure_state(new_state)

        penalty = ConstantReward(self.reward.stateaction_space, -penalty_level,
                                 reward_condition=penalty_condition)

        self.reward += penalty


class SafetySimulation(ModelLearningSimulation):
    def __init__(self, name, max_samples, gamma_optimistic, gamma_cautious,
                 lambda_cautious, shape, every, x_seed=None, y_seed=None,):

        if x_seed is None:
            self.x_seed = np.array([0.38, 0.54])
        else:
            self.x_seed = x_seed

        if y_seed is None:
            self.y_seed = np.array([.8])
        else:
            self.y_seed = y_seed

        dynamics_parameters = {
            'shape': shape
        }
        self.env = PenalizedSlip(dynamics_parameters=dynamics_parameters)

        self.ground_truth = SafetyTruth(self.env)
        self.ground_truth.from_vibly_file(
            Path(__file__).parent.parent.parent / 'data' / 'ground_truth' /
                'from_vibly' / 'slip_map.pickle'
        )

        # TODO Train this intsead
        self.hyperparameters = {
            'outputscale_prior': (0.4, 2),
            'lengthscale_prior': (0.2, 0.1),
            'noise_prior': (0.001, 0.002)
        }
        self.agent = SafetyLearner(
            env=self.env,
            gamma_optimistic=gamma_optimistic,
            gamma_cautious=gamma_cautious,
            lambda_cautious=lambda_cautious,
            x_seed=self.x_seed,
            y_seed=self.y_seed,
            gp_params=self.hyperparameters
        )
        plotters = {
            'DetailedSafety': DetailedSafetyPlotter(self.agent,
                                                    self.ground_truth)
        }

        output_directory = Path(__file__).parent.resolve()
        # building the super class (this class)
        super(SafetySimulation, self).__init__(output_directory, name,
                                               plotters)

        self.max_samples = max_samples
        self.every = every

    def get_models_to_save(self):
        return {'safety_values': self.agent.safety_model}

    def load_models(self, skip_local=False):
        model_name = list(self.get_models_to_save().keys())[0]
        if not skip_local:
            load_path = self.local_models_path / model_name
        else:
            load_path = self.models_path / model_name
        self.agent.value_model = SafetyMeasure.load(load_path, self.env,
                                                    self.x_seed, self.y_seed)

    def run_optim(self):
        train_x, train_y = self.ground_truth.get_training_examples(
            n_examples=2000,
            from_viable=True,
            from_failure=False
        )
        self.agent.fit_models(train_x, train_y, epochs=20)

    # def get_models_to_save(self):
    #     return {'q_values': self.agent.Q_model}

    # def load_models(self, skip_local=False):
    #     model_name = list(self.get_models_to_save().keys())[0]
    #     if not skip_local:
    #         load_path = self.local_models_path / model_name
    #     else:
    #         load_path = self.models_path / model_name
    #     self.agent.value_model = GPQLearning.load(load_path, self.env,
    #                                               self.x_seed, self.y_seed)

    def run(self):
        n_samples = 0
        self.save_figs(prefix='0')

        # train hyperparameters
        # train_x, train_y = self.ground_truth.get_training_examples()

        # experiment parameters
        horizon = 500  # max traj length before resetting agent
        while n_samples < self.max_samples:
            failed = self.agent.failed
            n_steps = 0
            while not failed and n_steps < horizon:
                n_samples += 1
                n_steps += 1
                old_state = self.agent.state
                new_state, reward, failed = self.agent.step()
                action = self.agent.last_action
                # if n_samples > 300:
                #     self.agent.greed *= (n_samples - 300) / (n_samples - 299)

                self.on_run_iteration(n_samples, old_state, action, new_state,
                                      reward, failed)

                if n_samples >= self.max_samples:
                    break
            self.agent.reset()

    # do some stuff to print useful information

    def on_run_iteration(self, n_samples, old_state, action, new_state,
                         reward, failed):
        super(SafetySimulation, self).on_run_iteration(
            old_state, action, new_state, reward, failed
        )
        print(f'Step {n_samples}/{self.max_samples} - {old_state} '
              f' -> {action} -> {new_state} ({failed})')
        if n_samples % self.every == 0:
            self.save_figs(prefix=f'{n_samples}')
    # def on_run_iteration(self, n_samples, *args, **kwargs):
    #     super(PenalizedSimulation, self).on_run_iteration(*args, **kwargs)

    #     print(f'Iteration {n_samples}/{self.max_samples}: {self.agent.greed}')
    #     if n_samples % self.every == 0:
    #         self.save_figs(prefix=f'{n_samples}')


if __name__ == '__main__':
    sim = SafetySimulation(
        name='2000',
        max_samples=200,
        gamma_optimistic=0.5,
        gamma_cautious=0.6,
        lambda_cautious=0.01,
        shape=(201, 151),
        every=25
    )
    sim.set_seed(0)

    sim.run()
    sim.save_models()