import matplotlib.pyplot as plt

from edge.model.safety_models import MaternSafety

from ..subplotter import SafetyMeasureSubplotter, SafetyTruthSubplotter,\
    SampleSubplotter
from ..colors import corl_colors


class Plotter:
    def __init__(self, agent):
        self.agent = agent

    def get_figure(self):
        raise NotImplementedError

    def on_run_iteration(self):
        raise NotImplementedError


class CoRLPlotter(Plotter):
    def __init__(self, agent, ground_truth=None):
        super(CoRLPlotter, self).__init__(agent)

        for model in agent.models:
            if isinstance(model, MaternSafety):
                self.safety_subplotter = SafetyMeasureSubplotter(
                    agent,
                    model,
                    corl_colors
                )
                break
        self.sample_subplotter = SampleSubplotter(corl_colors)
        self.truth_subplotter = SafetyTruthSubplotter(ground_truth,
                                                      corl_colors)

    def get_figure(self):
        figure = plt.figure(constrained_layout=True, figsize=(5.5, 4.8))
        gs = figure.add_gridspec(1, 2, width_ratios=[3, 1])

        ax_Q = figure.add_subplot(gs[0, 0])
        ax_S = figure.add_subplot(gs[0, 1], sharey=ax_Q)

        ax_Q.tick_params(direction='in', top=True, right=True)
        ax_S.tick_params(direction='in', left=False)

        Q_optimistic, Q_cautious, S_optimistic = self.get_subplotters_params()
        self.safety_subplotter.draw_on_axs(ax_Q, ax_S, Q_optimistic,
                                           Q_cautious, S_optimistic)
        self.sample_subplotter.draw_on_axs(ax_Q)
        self.truth_subplotter.draw_on_axs(ax_Q, ax_S)

        plt.title('Safety measure')
        return figure

    def get_subplotters_params(self):
        Q_optimistic, Q_cautious = self.safety_subplotter.model.level_set(
            state=None,
            lambda_threshold=[
                0,
                self.agent.lambda_cautious
            ],
            gamma_threshold=[
                self.agent.gamma_optimistic,
                self.agent.gamma_cautious
            ],
            return_covar=False
        )
        Q_optimistic = Q_optimistic.reshape(
            self.agent.env.stateaction_space.shape
        ).astype(float)
        Q_cautious = Q_cautious.reshape(
            self.agent.env.stateaction_space.shape
        ).astype(float)
        # The action axis is 1 because we can only plot 2D Stateaction spaces
        S_optimistic = Q_optimistic.mean(axis=1)
        return Q_optimistic, Q_cautious, S_optimistic

    def on_run_iteration(self, state, action, new_state, reward, failed):
        self.sample_subplotter.incur_sample(state, action, failed)
