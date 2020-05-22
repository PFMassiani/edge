from pathlib import Path
import numpy as np

from edge import GPModelLearningSimulation
from edge.envs import Hovership
from edge.agent import ConstrainedQLearner
from edge.reward import AffineReward
from edge.model.safety_models import SafetyTruth
from edge.model.value_models import GPQLearning
from edge.graphics.plotter import QValuePlotter


class LowGoalHovership(Hovership):
    def __init__(self, dynamics_parameters=None):
        super(LowGoalHovership, self).__init__(
            dynamics_parameters=dynamics_parameters
        )

        reward = AffineReward(self.stateaction_space, [(5,-5), (0, 0)])
        self.reward = reward


class ConstrainedSimulation(GPModelLearningSimulation):
    def __init__(self, name, max_samples, step_size, discount_rate,
                 x_seed, y_seed,
                 shape, every):
        dynamics_parameters = {
            'shape':shape
        }
        self.env = LowGoalHovership(dynamics_parameters=dynamics_parameters)

        self.ground_truth = SafetyTruth(self.env)
        self.ground_truth.from_vibly_file(
            Path(__file__).parent.parent.parent / 'data' / 'ground_truth' /
                'from_vibly' / 'hover_map.pickle'
        )

        self.hyperparameters = {
            'outputscale_prior': (0.4, 2),
            'lengthscale_prior': (0.2, 0.2),
            'noise_prior': (0.001, 0.002)
        }
        self.agent = ConstrainedQLearner(self.env, self.ground_truth,
                                         0.1, step_size, discount_rate,
                                         safety_threshold=0.05,
                                         x_seed=x_seed, y_seed=y_seed,
                                         gp_params=self.hyperparameters)

        plotters = {
            'Q-Values': QValuePlotter(self.agent, self.agent.safety_measure)
        }

        output_directory = Path(__file__).parent.resolve()
        super(ConstrainedSimulation, self).__init__(output_directory, name,
                                                    plotters)

        self.max_samples = max_samples
        self.every = every

    def get_models_to_save(self):
        return {'q_values': self.agent.Q_model}

    def load_models(self, skip_local=False):
        model_name = list(self.get_models_to_save().keys())[0]
        if not skip_local:
            local_path = self.local_models_path / model_name
            if local_path.exists():
                load_path = local_path
        else:
            load_path = self.models_path / model_name
        self.agent.value_model = GPQLearning.load(load_path)

    def run(self):
        n_samples = 0
        self.save_figs(prefix='0')
        while n_samples < self.max_samples:
            failed = self.agent.failed
            n_steps = 0
            while not failed and n_steps < 50:
                n_samples += 1
                n_steps += 1
                old_state = self.agent.state
                new_state, reward, failed = self.agent.step()
                action = self.agent.last_action

                self.on_run_iteration(n_samples, old_state, action, new_state,
                                      reward, failed)

                if n_samples >= self.max_samples:
                    break
            reset_state = self.agent.get_random_safe_state()
            self.agent.reset(reset_state)

    def on_run_iteration(self, n_samples, *args, **kwargs):
        super(ConstrainedSimulation, self).on_run_iteration(*args, **kwargs)

        print(f'Iteration {n_samples}/{self.max_samples}')
        if n_samples % self.every == 0:
            self.save_figs(prefix=f'{n_samples}')


if __name__ == '__main__':
    sim = ConstrainedSimulation(
        name='constrained',
        max_samples=1000,
        step_size=0.6,
        discount_rate=0.9,
        x_seed=np.array([1.45, 0.7]),
        y_seed=np.array([1]),
        shape=(201, 151),
        every=50
    )
    sim.set_seed(0)

    sim.run()
    sim.save_models()