import numpy as np

from . import Subplotter


class EpisodicScalarSubplotter(Subplotter):
    def __init__(self, color, name, ylabel):
        super(EpisodicScalarSubplotter, self).__init__(color)
        self.episode_scalars = []
        self.aggregated_scalars = []

        self.color = color
        self.name = name
        self.ylabel = ylabel

    def draw_on_axs(self, ax_Scal):
        ax_Scal.plot(
            self.aggregated_scalars,
            # color=self.color,
            label=self.name
        )

        ax_Scal.set_xlabel('Episode')
        ax_Scal.set_ylabel(self.ylabel)

        return ax_Scal

    def on_run_iteration(self, state, action, new_state, reward, failed, done,
                         *args, **kwargs):
        self.episode_scalars.append(self._extract_scalar(
            state, action, new_state, reward, failed, done, *args, **kwargs
        ))
        if done:
            average_reward = self._aggregate_scalars(self.episode_scalars)
            self.aggregated_scalars.append(average_reward)
            self.episode_scalars = []

    def _extract_scalar(self, state, action, new_state, reward, failed, done,
                         *args, **kwargs):
        raise NotImplementedError

    def _aggregate_scalars(self, episode_scalars):
        raise NotImplementedError


class EpisodicRewardSubplotter(EpisodicScalarSubplotter):
    def __init__(self, color, name=None):
        super(EpisodicRewardSubplotter, self).__init__(
            color,
            name,
            'Reward'
        )

    def _extract_scalar(self, state, action, new_state, reward, failed, done,
                         *args, **kwargs):
        return reward

    def _aggregate_scalars(self, episode_scalars):
        return np.sum(episode_scalars)


class SmoothedEpisodicFailureSubplotter(EpisodicScalarSubplotter):
    def __init__(self, window_size, color, name=None, padding_value=1):
        super(SmoothedEpisodicFailureSubplotter, self).__init__(
            color,
            name,
            'Smoothed failure'
        )
        self.window_size = window_size
        self.padding_value = padding_value

    def _extract_scalar(self, state, action, new_state, reward, failed, done,
                         *args, **kwargs):
        return int(failed)

    def _aggregate_scalars(self, episode_scalars):
        has_failed = max(self.episode_scalars)
        history = self.aggregated_scalars[-self.window_size:]
        padding_length = self.window_size - len(history)
        padding = [self.padding_value for _ in range(padding_length)]
        values_to_average = np.concatenate((history, padding, [has_failed]))
        return np.mean(values_to_average)