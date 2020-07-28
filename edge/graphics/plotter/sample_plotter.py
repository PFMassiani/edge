import matplotlib.pyplot as plt

from . import Plotter
from ..subplotter import SampleSubplotter, SafetyTruthSubplotter
from ..colors import corl_colors


class SamplePlotter(Plotter):
    def __init__(self, agent, ground_truth=None):
        super(SamplePlotter, self).__init__(agent)
        self.sample_subplotter = SampleSubplotter(corl_colors)
        if ground_truth is not None:
            self.truth_subplotter = SafetyTruthSubplotter(ground_truth,
                                                          corl_colors)

    def get_figure(self):
        figure = plt.figure(figsize=(5.5, 4.8))
        ax = figure.add_subplot()

        ax.tick_params(direction='in', top=True, right=True)

        if self.truth_subplotter is not None:
            self.truth_subplotter.draw_on_axs(ax_Q=ax, ax_S=None)
        self.sample_subplotter.draw_on_axs(ax)

        ax.set_xlabel('action space $A$')
        ax.set_ylabel('state space $S$')
        frame_width_x = self.agent.env.action_space[-1] * .03
        ax.set_xlim((self.agent.env.action_space[0] - frame_width_x,
                       self.agent.env.action_space[-1] + frame_width_x))

        frame_width_y = self.agent.env.state_space[-1] * .03
        ax.set_ylim((self.agent.env.state_space[0] - frame_width_y,
                       self.agent.env.state_space[-1] + frame_width_y))
        plt.title('Samples')

        return figure

    def flush_samples(self):
        self.sample_subplotter.flush_samples()

    def on_run_iteration(self, state, action, new_state, reward, failed):
        self.sample_subplotter.incur_sample(state, action, failed)