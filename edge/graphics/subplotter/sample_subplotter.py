from . import Subplotter
import numpy as np


class SampleSubplotter(Subplotter):
    def __init__(self, colors):
        super(SampleSubplotter, self).__init__(colors)
        self.failed_samples = []
        self.unfailed_samples = []
        self.failed_colors = []
        self.unfailed_colors = []

    def incur_sample(self, state, action, failed, color=None):
        if color is None:
            color = [0.9, 0.3, 0.3]
        # States and actions are stored in np arrays of shape (1,) (since we
        # are plotting them)
        if failed:
            self.failed_samples.append((state[0], action[0]))
            self.failed_colors.append(color)
        else:
            self.unfailed_samples.append((state[0], action[0]))
            self.unfailed_colors.append(color)

    def flush_samples(self):
        self.failed_samples = []
        self.unfailed_samples = []
        self.failed_colors = []
        self.unfailed_colors = []

    def ensure_samples_in_at_least_one(self, *datasets):
        dataset = np.unique(
            np.vstack(datasets),
            axis=0
        )

        def is_in_dataset(to_check):
            return [np.isclose(x, dataset).all(axis=1).any() for x in to_check]
        failed_in = is_in_dataset(self.failed_samples)
        unfailed_in = is_in_dataset(self.unfailed_samples)

        def filter_list(to_filter, keep_bools):
            return [x for x, keep in zip(to_filter, keep_bools) if keep]
        self.failed_samples = filter_list(self.failed_samples, failed_in)
        self.unfailed_samples = filter_list(self.unfailed_samples, unfailed_in)
        self.failed_colors = filter_list(self.failed_colors, failed_in)
        self.unfailed_colors = filter_list(self.unfailed_colors, unfailed_in)

    def draw_on_axs(self, ax_Q):
        def scatter_stateactions(stateactions, colors, marker):
            states, actions = zip(*stateactions)
            ax_Q.scatter(
                actions,
                states,
                color=colors,
                s=30,
                marker=marker,
                edgecolors='none'
            )

        if len(self.failed_samples) > 0:
            scatter_stateactions(self.failed_samples, self.failed_colors, 'x')
        if len(self.unfailed_samples) > 0:
            scatter_stateactions(self.unfailed_samples, self.unfailed_colors, '.')
