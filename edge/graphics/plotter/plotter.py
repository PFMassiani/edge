import matplotlib.pyplot as plt

from edge.models import SafetyModel
from edge.utils import dynamically_import

from ..subplotter import SafetyMeasureSubplotter, SafetyTruthSubplotter
from ..colors import corl_colors


class Plotter:
    def __init__(self, agent):
        self.agent = agent

    def get_figure(self):
        raise NotImplementedError


class CoRLPlotter(Plotter):
    def __init__(self, agent, ground_truth):
        super(CoRLPlotter, self).__init__(agent)

        self.subplotters = {}
        for model in agent.models:
            if isinstance(model, SafetyModel):
                self.subplotters['safety'] = SafetyMeasureSubplotter(
                    agent,
                    model,
                    corl_colors
                )
        self.subplotters['truth'] = SafetyTruthSubplotter(ground_truth)

    def get_figure(self):
        figure = plt.figure(constrained_layout=True, figsize=(5.5, 4.8))
        gs = figure.add_gridspec(1, 2, width_ratios=[3, 1])

        ax_Q = figure.add_subplot(gs[0, 0])
        ax_S = figure.add_subplot(gs[0, 0], sharey=ax_Q)

        ax_Q.tick_params(direction='in', top=True, right=True)
        ax_S.tick_params(direction='in', left=False)

        self.subplotters['safety'].draw_on_axs(ax_Q, ax_S)
        self.subplotters['truth'].draw_on_axs(ax_Q, ax_S)

        figure.title('Safety measure')
        return figure
