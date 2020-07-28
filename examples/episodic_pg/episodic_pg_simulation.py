from pathlib import Path
import numpy as np

from edge import ModelLearningSimulation
from edge.graphics.plotter import SamplePlotter
from edge.model.safety_models import SafetyTruth

from episodic_pg_parameterization import LowGoalSlip, PGOptimizer


def affine_interpolation(t, start, end):
    return start + (end - start) * t


def identity_or_duplicated_value(possible_tuple):
    if isinstance(possible_tuple, tuple):
        return possible_tuple
    else:
        return possible_tuple, possible_tuple


class EpisodicPGSimulation(ModelLearningSimulation):
    def __init__(self, name, n_episodes, episode_max_steps,
                 discount_rate, step_size, features_function, n_features, initial_weight, initial_var,
                 shape):
        dynamics_parameters = {
            'shape': shape
        }
        self.env = LowGoalSlip(dynamics_parameters=dynamics_parameters)

        self.agent = PGOptimizer(
            env=self.env,
            discount_rate=discount_rate,
            step_size=step_size,
            features_function=features_function,
            n_features=n_features,
            initial_weight=initial_weight,
            initial_var=initial_var
        )

        self.ground_truth = SafetyTruth(self.env)
        self.ground_truth.from_vibly_file(
            Path(__file__).parent.parent.parent / 'data' / 'ground_truth' /
            'from_vibly' / 'slip_map.pickle'
        )

        plotters = {
            'Samples': SamplePlotter(self.agent, self.ground_truth)
        }

        output_directory = Path(__file__).parent.resolve()
        super(EpisodicPGSimulation, self).__init__(output_directory, name,
                                                   plotters)

        self.n_episodes = n_episodes
        self.episode_max_steps = episode_max_steps

    def get_models_to_save(self):
        return {}  # TODO: so far, the models are not saved

    def load_models(self, skip_local=False):
        pass  # TODO

    def run_episode(self, n_episode):
        n_steps = 0
        episode = []
        while n_steps < self.episode_max_steps:
            n_steps += 1
            old_state = self.agent.state
            new_state, reward, failed = self.agent.step()
            action = self.agent.last_action
            step = {
                'state': old_state,
                'action': action,
                'new_state': new_state,
                'reward': reward,
                'failed': failed
            }
            episode.append(step)
            self.on_run_iteration(n_episode, n_steps, old_state, action, new_state,
                                  reward, failed)
            if failed:
                break
        return episode

    def run(self):
        n_episode = 0
        self.save_figs(prefix='Ep0')
        while n_episode < self.n_episodes:
            n_episode += 1
            self.agent.reset(np.array([0.4]))
            episode = self.run_episode(n_episode)
            self.agent.update_models(episode)
            self.save_figs(prefix=f'Ep{n_episode}')
            self.on_episode_iteration()
        print('Done.')

    def on_run_iteration(self, n_episode, n_steps, *args, **kwargs):
        super(EpisodicPGSimulation, self).on_run_iteration(*args, **kwargs)
        print(f'Episode {n_episode} - Step {n_steps}')
        print(self.agent.policy.actions_density)

    def on_episode_iteration(self):
        self.plotters['Samples'].flush_samples()


if __name__ == '__main__':
    def features_function(x):
        return np.concatenate([[1], x])
    n_features = 2

    sim = EpisodicPGSimulation(
        name='test',
        n_episodes=100,
        episode_max_steps=50,
        discount_rate=0.8,
        step_size=0.6,
        features_function=features_function,
        n_features=n_features,
        initial_weight=np.array([1, -1], dtype=float),
        initial_var=0.01,
        shape=(201, 201)
    )
    sim.set_seed(0)

    sim.run()
    sim.save_models()