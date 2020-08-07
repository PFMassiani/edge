from matplotlib import pyplot as plt

from . import Plotter
from ..subplotter import EpisodicRewardSubplotter, \
    SmoothedEpisodicFailureSubplotter
from ..colors import corl20_colors as colors


class RewardFailurePlotter(Plotter):
    def __init__(self, agents_names, window_size, padding_value=1):
        super(RewardFailurePlotter, self).__init__(agent=None)

        self.agents_names = agents_names
        n_agents = len(self.agents_names)
        self.reward_subplotter = {
            aname: EpisodicRewardSubplotter(
                color=colors.dark_blue,
                name=aname
            ) for aname in self.agents_names
        }
        self.failure_subplotter = {
            aname: SmoothedEpisodicFailureSubplotter(
                window_size=window_size,
                color=colors.episodic_failure_colors(t),
                name=aname,
                padding_value=padding_value
            ) for aname, t in zip(self.agents_names, range(n_agents))
        }
        self.has_data = {aname: False for aname in self.agents_names}

    def on_run_iteration(self, aname, *args, **kwargs):
        self.reward_subplotter[aname].on_run_iteration(*args, **kwargs)
        self.failure_subplotter[aname].on_run_iteration(*args, **kwargs)
        self.has_data[aname] = True

    def get_figure(self):
        figure = plt.figure(constrained_layout=True, figsize=(4, 4))
        gs = figure.add_gridspec(2, 1)

        ax_R = figure.add_subplot(gs[0, 0])
        ax_F = figure.add_subplot(gs[1, 0], sharex=ax_R)

        ax_R.tick_params(direction='in', top=True, right=True)
        ax_F.tick_params(direction='in', top=True, right=True)

        ax_R.grid(True)
        ax_F.grid(True)

        for aname in self.agents_names:
            if self.has_data[aname]:
                self.reward_subplotter[aname].draw_on_axs(ax_R)
                self.failure_subplotter[aname].draw_on_axs(ax_F)

        ax_R.legend()
        ax_F.legend()

        return figure