from matplotlib import pyplot as plt

from . import Plotter
from ..subplotter import QValueSubplotter, SampleSubplotter, SafetyTruthSubplotter, SafetyMeasureSubplotter
from ..colors import corl_colors


class QValueAndSafetyPlotter(Plotter):
    def __init__(self, agent, safety_truth=None):
        super(QValueAndSafetyPlotter, self).__init__(agent)

        self.q_value_subplotter = QValueSubplotter(agent, corl_colors, write_values=False)
        self.safety_subplotter = SafetyMeasureSubplotter(agent, corl_colors, fill=False, plot_optimistic=False)
        self.sample_subplotter = SampleSubplotter(corl_colors)
        if safety_truth is not None:
            self.safety_truth_subplotter = SafetyTruthSubplotter(safety_truth,
                                                                 corl_colors)
        else:
            self.safety_truth_subplotter = None

    def get_figure(self):
        figure = plt.figure(constrained_layout=True, figsize=(5.5, 4.8))
        gs = figure.add_gridspec(1, 2, width_ratios=[3, 1])

        ax_Q = figure.add_subplot(gs[0, 0])
        ax_S = figure.add_subplot(gs[0, 1], sharey=ax_Q)

        ax_Q.tick_params(direction='in', top=True, right=True)
        ax_S.tick_params(direction='in', left=False)

        Q_values, Q_optimistic, Q_cautious, S_cautious = self.get_subplotters_params()

        if self.safety_truth_subplotter is not None:
            self.safety_truth_subplotter.draw_on_axs(ax_Q, ax_S)
        q_values_image = self.q_value_subplotter.draw_on_axs(ax_Q, Q_values)
        self.safety_subplotter.draw_on_axs(ax_Q, ax_S, Q_optimistic,
                                           Q_cautious, S_cautious)
        self.sample_subplotter.draw_on_axs(ax_Q)

        figure.colorbar(q_values_image, ax=ax_S, location='right')

        # plt.title('Q-Values map with viable set estimate')
        return figure

    def get_subplotters_params(self):
        Q_values = self.agent.Q_model[:, :].reshape(
            self.agent.env.stateaction_space.shape
        )

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
        S_cautious = Q_cautious.mean(axis=1)

        return Q_values, Q_optimistic, Q_cautious, S_cautious

    def on_run_iteration(self, state, action, new_state, reward, failed):
        self.sample_subplotter.incur_sample(state, action, failed)