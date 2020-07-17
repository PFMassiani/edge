from matplotlib import pyplot as plt

from . import Plotter
from ..subplotter import QValueSubplotter, SampleSubplotter,\
    SafetyTruthSubplotter
from ..colors import corl_colors


class QValuePlotter(Plotter):
    def __init__(self, agent, safety_truth=None, write_values=False):
        super(QValuePlotter, self).__init__(agent)

        self.q_value_subplotter = QValueSubplotter(agent, corl_colors, write_values)
        if safety_truth is not None:
            self.safety_truth_subplotter = SafetyTruthSubplotter(safety_truth,
                                                                 corl_colors)
        else:
            self.safety_truth_subplotter = None

    def get_figure(self):
        figure = plt.figure(constrained_layout=True,figsize=(4, 4))

        ax_Q = figure.add_subplot()
        ax_Q.tick_params(direction='in', top=True, right=True)

        Q_values = self.get_subplotters_params()

        if self.safety_truth_subplotter is not None:
            self.safety_truth_subplotter.draw_on_axs(ax_Q)
        q_values_image = self.q_value_subplotter.draw_on_axs(ax_Q, Q_values)

        figure.colorbar(q_values_image, ax=ax_Q, location='right')

        plt.title('Q-Values map')
        return figure

    def get_subplotters_params(self):
        Q_values = self.agent.Q_model[:, :].reshape(
            self.agent.env.stateaction_space.shape
        )
        return Q_values

    def on_run_iteration(self, state, action, new_state, reward, failed):
        pass