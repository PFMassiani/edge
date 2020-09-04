from pathlib import Path

from edge import Simulation, ModelLearningSimulation
from edge.agent import DiscreteQLearner
from edge.model.safety_models import SafetyTruth
from edge.graphics.plotter import DualityPlotter, DiscreteQValuePlotter
from edge.model.value_models import QLearning
from edge.envs import DiscreteHovership
from edge.reward import AffineReward, ConstantReward


class LowGoalHovership(DiscreteHovership):
    def __init__(self, dynamics_parameters=None):
        super(LowGoalHovership, self).__init__(
            random_start=True,
            dynamics_parameters=dynamics_parameters,
            reward_done_threshold=None,
            steps_done_threshold=15,
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


class DualitySimulation(ModelLearningSimulation):
    def __init__(self, output_directory, name, penalty, greed, step_size, discount_rate, max_samples, glie_start=None):
        super(DualitySimulation, self).__init__(output_directory, name,
                                                    None)
        dynamics_parameters = {
            'ground_gravity': 1,
            'gravity_gradient': 1,
            'max_thrust': 4,
            'max_altitude': 10,
            'minimum_gravity_altitude': 9,
            'maximum_gravity_altitude': 3
        }
        if penalty is None:
            self.env = LowGoalHovership(dynamics_parameters=dynamics_parameters)
        else:
            self.env = PenalizedHovership(penalty_level=penalty,
                                          dynamics_parameters=dynamics_parameters)

        self.ground_truth = self.get_ground_truth()
        if penalty is None:
            constraint = self.ground_truth
            safety_threshold = 0
        else:
            constraint = None
            safety_threshold = None
        self.agent = DiscreteQLearner(self.env, greed, step_size, discount_rate,
                                      constraint=constraint,
                                      safety_threshold=safety_threshold)

        plotters = {
            'Q-Values': DiscreteQValuePlotter(self.agent, self.ground_truth,
                                      vmin=-5, vmax=5)
        }
        self.plotters = plotters

        self.max_samples = max_samples
        self.glie_start = glie_start

    def get_models_to_save(self):
        return {'q_values': self.agent.Q_model}

    def load_models(self, skip_local=False):
        model_name = list(self.get_models_to_save().keys())[0]
        if not skip_local:
            load_path = self.local_models_path / model_name

        else:
            load_path = self.models_path / model_name
        self.agent.value_model = QLearning.load(load_path)

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
        while n_samples < self.max_samples:
            reset_state = self.agent.get_random_safe_state()
            self.agent.reset(reset_state)
            failed = self.agent.failed
            n_steps = 0
            while not failed and n_steps < 50:
                n_samples += 1
                n_steps += 1
                old_state = self.agent.state
                new_state, reward, failed = self.agent.step()
                action = self.agent.last_action
                if self.glie_start is not None and n_samples > self.glie_start:
                    self.agent.decrease_step_size()

                self.on_run_iteration(n_samples, old_state, action, new_state,
                                      reward, failed)

                if n_samples >= self.max_samples or old_state == new_state:
                    break

        self.save_figs(prefix=f'{self.name}_final')

    def on_run_iteration(self, n_samples, old_state, action, new_state,
                                      reward, failed):
        super(DualitySimulation, self).on_run_iteration(old_state, action,
                                                          new_state, reward,
                                                          failed)

        print(f'Iteration {n_samples}/{self.max_samples}: {self.agent.greed} | '
              f'{old_state} -> {action} -> {new_state} ({reward})')


class DualitySimulationsGroup(Simulation):
    def __init__(self, name, penalties, greed, step_size, discount_rate, max_samples, glie_start=None):
        output_directory = Path(__file__).parent.resolve()
        super(DualitySimulationsGroup, self).__init__(output_directory, name, None)
        self.constrained_sim = DualitySimulation(
            output_directory=self.output_directory / 'simulations',
            name='constrained',
            penalty=None,
            greed=greed,
            step_size=step_size,
            discount_rate=discount_rate,
            max_samples=max_samples,
            glie_start=glie_start
        )
        self.penalized_sims = [
            DualitySimulation(
                output_directory=self.output_directory / 'simulations',
                name=f'penalized_{p}',
                penalty=p,
                greed=greed,
                step_size=step_size,
                discount_rate=discount_rate,
                max_samples=max_samples,
                glie_start=glie_start
            ) for p in penalties
        ]

        self.plotters = {
            'values': DualityPlotter(
                penalties,
                self.constrained_sim.agent,
                *[pen_sim.agent for pen_sim in self.penalized_sims])
        }

    def run(self):
        self.constrained_sim.run()
        for pen_sim in self.penalized_sims:
            pen_sim.run()
        self.save_figs(prefix='final')

    def save_models(self, globally=False, locally=True):
        self.constrained_sim.save_models(globally, locally)
        for pen_sim in self.penalized_sims:
            pen_sim.save_models(globally, locally)


if __name__ == '__main__':
    sim_group = DualitySimulationsGroup(
        name='thesis',
        penalties=[0, 5, 15, 200],
        greed=0.3,
        step_size=0.6,
        discount_rate=0.3,
        max_samples=10000,
        glie_start=None
    )
    sim_group.set_seed(0)

    sim_group.run()
    sim_group.save_models()