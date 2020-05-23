from pathlib import Path
import numpy as np

from edge import ModelLearningSimulation
from edge.envs import DiscreteHovership
from edge.agent import DiscreteQLearner
from edge.reward import AffineReward
from edge.model.safety_models import SafetyTruth
from edge.model.value_models import GPQLearning
from edge.graphics.plotter import QValuePlotter


class LowGoalHovership(DiscreteHovership):
    def __init__(self, dynamics_parameters=None):
        super(LowGoalHovership, self).__init__(
            random_start=True,
            dynamics_parameters=dynamics_parameters
        )

        reward = AffineReward(self.stateaction_space, [(5,-5), (0, 0)])
        self.reward = reward


class ConstrainedSimulation(ModelLearningSimulation):
    def __init__(self, name, max_samples, greed, step_size, discount_rate,
                 every):
        dynamics_parameters = {
        }
        self.env = LowGoalHovership(dynamics_parameters=dynamics_parameters)

        # TODO implement Ground Truth computation
        self.ground_truth = SafetyTruth(self.env)
        self.ground_truth.from_vibly_file(
            Path(__file__).parent.parent.parent / 'data' / 'ground_truth' /
                'from_vibly' / 'hover_map.pickle'
        )

        self.agent = DiscreteQLearner(self.env, greed, step_size, discount_rate,
                                      constraint=self.ground_truth,
                                      safety_threshold=0.05)

        plotters = {
            'Q-Values': QValuePlotter(self.agent, self.agent.constraint)
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
            load_path = self.local_models_path / model_name

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
                # if n_samples > 300:
                #     self.agent.greed *= (n_samples - 300) / (n_samples - 299)

                self.on_run_iteration(n_samples, old_state, action, new_state,
                                      reward, failed)

                if n_samples >= self.max_samples:
                    break
            reset_state = self.agent.get_random_safe_state()
            self.agent.reset(reset_state)

    def on_run_iteration(self, n_samples, *args, **kwargs):
        super(ConstrainedSimulation, self).on_run_iteration(*args, **kwargs)

        print(f'Iteration {n_samples}/{self.max_samples}: {self.agent.greed}')
        if n_samples % self.every == 0:
            self.save_figs(prefix=f'{n_samples}')


if __name__ == '__main__':
    sim = ConstrainedSimulation(
        name='constrained',
        max_samples=1000,
        greed=0.1,
        step_size=0.6,
        discount_rate=0.9,
        every=50
    )
    sim.set_seed(0)

    sim.run()
    sim.save_models()