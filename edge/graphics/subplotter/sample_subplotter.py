from . import Subplotter


class SampleSubplotter(Subplotter):
    def __init__(self, colors):
        super(SampleSubplotter, self).__init__(colors)
        self.failed_samples = []
        self.unfailed_samples = []

    def incur_sample(self, state, action, failed):
        # States and actions are stored in np arrays of shape (1,) (since we
        # are plotting them)
        if failed:
            self.failed_samples.append((state[0], action[0]))
        else:
            self.unfailed_samples.append((state[0], action[0]))

    def flush_samples(self):
        self.failed_samples = []
        self.unfailed_samples = []

    def draw_on_axs(self, ax_Q):
        def scatter_stateactions(stateactions, marker):
            states, actions = zip(*stateactions)
            ax_Q.scatter(
                actions,
                states,
                facecolors=[[0.9, 0.3, 0.3]],
                s=30,
                marker=marker,
                edgecolors='none'
            )

        if len(self.failed_samples) > 0:
            scatter_stateactions(self.failed_samples, 'x')
        if len(self.unfailed_samples) > 0:
            scatter_stateactions(self.unfailed_samples, '.')
