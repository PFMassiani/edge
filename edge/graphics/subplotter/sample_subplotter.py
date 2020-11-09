from . import Subplotter
import numpy as np


def masked(to_mask, mask):
    return [item for item, keep in zip(to_mask, mask) if keep]


class SampleSubplotter(Subplotter):
    def __init__(self, colors):
        super(SampleSubplotter, self).__init__(colors)
        self.failed_samples = []
        self.unfailed_samples = []
        self.failed_colors = []
        self.unfailed_colors = []
        self.failed_markers = []
        self.unfailed_markers = []

    def incur_sample(self, state, action, failed, color=None, marker=None):
        if color is None:
            color = [0.9, 0.3, 0.3]
        # States and actions are stored in np arrays of shape (1,) (since we
        # are plotting them)
        if failed:
            marker = marker if marker is not None else 'x'
            self.failed_samples.append((state[0], action[0]))
            self.failed_colors.append(color)
            self.failed_markers.append(marker)
        else:
            marker = marker if marker is not None else '.'
            self.unfailed_samples.append((state[0], action[0]))
            self.unfailed_colors.append(color)
            self.unfailed_markers.append(marker)

    def flush_samples(self):
        self.failed_samples = []
        self.unfailed_samples = []
        self.failed_colors = []
        self.unfailed_colors = []
        self.failed_markers = []
        self.unfailed_markers = []

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
        self.failed_markers = filter_list(self.failed_markers, failed_in)
        self.unfailed_markers = filter_list(self.unfailed_markers, failed_in)

    def draw_on_axs(self, ax_Q):
        def scatter_stateactions(stateactions, colors, markers):
            markers_set = set(markers)
            for marker in markers_set:
                fltr = [m == marker for m in markers]
                if any(fltr):
                    states, actions = zip(*masked(stateactions, fltr))
                    ax_Q.scatter(
                        actions,
                        states,
                        color=masked(colors, fltr),
                        s=30,
                        marker=marker,
                        edgecolors='none'
                    )

        if len(self.failed_samples) > 0:
            scatter_stateactions(self.failed_samples, self.failed_colors,
                                 self.failed_markers)
        if len(self.unfailed_samples) > 0:
            scatter_stateactions(self.unfailed_samples, self.unfailed_colors,
                                 self.unfailed_markers)
