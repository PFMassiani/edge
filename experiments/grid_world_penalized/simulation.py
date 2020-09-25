from pathlib import Path
import numpy as np

from edge import ModelLearningSimulation
from edge.envs import DiscreteHovership
from edge.agent import DiscreteQLearner
from edge.reward import AffineReward, ConstantReward
from edge.model.safety_models import SafetyTruth
from edge.model.value_models import GPQLearning
from edge.graphics.plotter import QValuePlotter


class LowGoalHovership(DiscreteHovership):
    def __init__(self, dynamics_parameters=None):
        super(LowGoalHovership, self).__init__(
            random_start=True,
            dynamics_parameters=dynamics_parameters
        )

        reward = AffineReward(self.stateaction_space, [(10,0), (0, 0)])
        self.reward = reward


class PenalizedHovership(LowGoalHovership):
    def __init__(self, penalty_level=100, dynamics_parameters=None):
        super(PenalizedHovership, self).__init__(dynamics_parameters)

        def penalty_condition(state, action, new_state, reward):
            return self.is_failure_state(new_state)

        penalty = ConstantReward(self.reward.stateaction_space, -penalty_level,
                                 reward_condition=penalty_condition)

        self.reward += penalty


class PenalizedSimulation(ModelLearningSimulation):
    def __init__(self, name, max_samples, greed, step_size, discount_rate,
                 penalty_level, every, glie_start):
        output_directory = Path(__file__).parent.resolve()
        super(PenalizedSimulation, self).__init__(output_directory, name,
                                                    None)
        dynamics_parameters = {
            'ground_gravity': 1,
            'gravity_gradient': 1,
            'max_thrust': 4,
            'max_altitude': 10,
            'minimum_gravity_altitude': 9,
            'maximum_gravity_altitude': 3
        }
        self.env = PenalizedHovership(penalty_level=penalty_level,
                                      dynamics_parameters=dynamics_parameters)

        self.ground_truth = self.get_ground_truth()

        self.agent = DiscreteQLearner(self.env, greed, step_size, discount_rate)

        plotters = {
            'Q-Values': QValuePlotter(self.agent, self.ground_truth)
        }

        self.plotters = plotters

        self.max_samples = max_samples
        self.every = every
        self.glie_start = glie_start

    def get_models_to_save(self):
        return {'q_values': self.agent.Q_model}

    def load_models(self, skip_local=False):
        model_name = list(self.get_models_to_save().keys())[0]
        if not skip_local:
            load_path = self.local_models_path / model_name

        else:
            load_path = self.models_path / model_name
        self.agent.value_model = GPQLearning.load(load_path)

    def get_ground_truth(self):
        self.ground_truth_path = self.local_models_path / 'safety_ground_truth.npz'
        load = self.ground_truth_path.exists()
        if load:
            try:
                ground_truth = SafetyTruth.load(
                    self.ground_truth_path, self.env
                )
            except ValueError:
                load = False
        if not load:
            ground_truth = SafetyTruth(self.env)
            ground_truth.compute()
            ground_truth.save(self.ground_truth_path)

        return ground_truth

    def run(self):
        n_samples = 0
        self.save_figs(prefix='0')
        done = False
        while (not done) and (n_samples < self.max_samples):
            reset_state = self.agent.get_random_safe_state()
            self.agent.reset(reset_state)
            failed = self.agent.failed
            n_steps = 0
            while not failed and n_steps < 50:
                n_samples += 1
                n_steps += 1
                old_state = self.agent.state
                new_state, reward, failed, done = self.agent.step()
                action = self.agent.last_action
                if self.glie_start is not None and n_samples > self.glie_start:
                    self.agent.greed *= (n_samples - self.glie_start) / (
                            n_samples - (self.glie_start - 1)
                    )

                self.on_run_iteration(n_samples, old_state, action, new_state,
                                      reward, failed)

                if n_samples >= self.max_samples or old_state == new_state:
                    break

        self.save_figs(prefix='final')

    def on_run_iteration(self, n_samples, old_state, action, new_state,
                                      reward, failed):
        # super(PenalizedSimulation, self).on_run_iteration(old_state, action,
        #                                                   new_state, reward,
        #                                                   failed)

        # print(f'Iteration {n_samples}/{self.max_samples}: {self.agent.greed} | '
        #       f'{old_state} -> {action} -> {new_state} ({reward})')
        if n_samples % self.every == 0:
            self.save_figs(prefix=f'{n_samples}')


if __name__ == '__main__':
    penalties = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 10000]
    for penalty in penalties:
        print('Running simulation with penalty', penalty, '...')
        sim = PenalizedSimulation(
            name='penalty_' + str(penalty),
            max_samples=10000,
            greed=0.1,
            step_size=0.6,
            discount_rate=0.9,
            penalty_level=penalty,
            every=1000,
            glie_start=7000
        )
        sim.set_seed(0)

        sim.run()
        sim.save_models()