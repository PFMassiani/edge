from pathlib import Path
import numpy as np

from edge import ModelLearningSimulation
from edge.envs import Slip
from edge.agent import QLearner
from edge.reward import AffineReward, ConstantReward
from edge.model.safety_models import SafetyTruth
from edge.model.value_models import GPQLearning
from edge.graphics.plotter import QValuePlotter


class LowGoalSlip(Slip):
    # * This goal incentivizes the agent to run fast
    def __init__(self, dynamics_parameters=None):
        super(LowGoalSlip, self).__init__(
            dynamics_parameters=dynamics_parameters,
            random_start=True
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


class PenalizedSimulation(ModelLearningSimulation):
    def __init__(self, name, max_samples, greed, step_size, discount_rate,
                 penalty_level, x_seed, y_seed,
                 shape, every):
        dynamics_parameters = {
            'shape': shape
        }
        self.env = PenalizedSlip(penalty_level=penalty_level,
                                 dynamics_parameters=dynamics_parameters)

        self.ground_truth = SafetyTruth(self.env)
        self.ground_truth.from_vibly_file(
            Path(__file__).parent.parent.parent / 'data' / 'ground_truth' /
                'from_vibly' / 'slip_map.pickle'
        )

        self.hyperparameters = {
            'outputscale_prior': (0.4, 2),
            'lengthscale_prior': (0.02, 0.01),
            'noise_prior': (0.001, 0.002)
        }
        self.x_seed = x_seed
        self.y_seed = y_seed
        self.agent = QLearner(self.env,
                              greed, step_size, discount_rate,
                              x_seed=self.x_seed, y_seed=self.y_seed,
                              gp_params=self.hyperparameters)

        plotters = {
            'Q-Values': QValuePlotter(self.agent, self.ground_truth)
        }

        # plotters = {}

        output_directory = Path(__file__).parent.resolve()
        super(PenalizedSimulation, self).__init__(output_directory, name,
                                                  plotters)

        self.max_samples = max_samples
        self.every = every

    def get_models_to_save(self):
        return {'q_values': self.agent.Q_model}

    def load_models(self, skip_local=False):
        model_name = list(self.get_models_to_save().keys())[0]
        if not skip_local:
            load_path = self.local_models_path / model_name
        else:
            load_path = self.models_path / model_name
        self.agent.value_model = GPQLearning.load(load_path,
                                                  self.env.stateaction_space,
                                                  self.x_seed, self.y_seed)

    def run(self):
        n_samples = 0
        self.save_figs(prefix='0')

        # train hyperparameters
        # train_x, train_y = self.ground_truth.get_training_examples()
        start_GLIE = round(0.3*self.max_samples)
        while n_samples < self.max_samples:
            failed = self.agent.failed
            n_steps = 0
            while not failed and n_steps < 50:
                n_samples += 1
                n_steps += 1
                old_state = self.agent.state
                new_state, reward, failed = self.agent.step()
                action = self.agent.last_action

                # * start reducing eps to converge to a greedy policy.
                if n_samples > start_GLIE:
                    self.agent.greed *= ((n_samples - start_GLIE) /
                                         (n_samples - start_GLIE + 1))

                self.on_run_iteration(n_samples, old_state, action, new_state,
                                      reward, failed)

                if n_samples >= self.max_samples:
                    break
            self.agent.reset()

    def on_run_iteration(self, n_samples, *args, **kwargs):
        super(PenalizedSimulation, self).on_run_iteration(*args, **kwargs)

        print(f'Iteration {n_samples}/{self.max_samples}: {self.agent.greed}')
        if n_samples % self.every == 0:
            self.save_figs(prefix=f'{n_samples}')


if __name__ == '__main__':
    sim = PenalizedSimulation(
        name='experiment',
        max_samples=500,
        greed=0.8,
        step_size=0.6,
        discount_rate=0.2,
        penalty_level=10,
        x_seed=np.array([0.4, 0.6]),
        y_seed=np.array([1]),
        shape=(201, 151),
        every=10
    )
    sim.set_seed(0)

    sim.run()
    sim.save_models()