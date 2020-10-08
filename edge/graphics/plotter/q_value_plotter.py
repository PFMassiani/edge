from matplotlib import pyplot as plt

from . import Plotter
from ..subplotter import QValueSubplotter, SampleSubplotter, \
    SafetyTruthSubplotter, DiscreteQValueSubplotter
from ..colors import corl_colors
from edge.envs import DiscreteHovership


class QValuePlotter(Plotter):
    def __init__(self, agent, safety_truth=None, write_values=False,
                 plot_samples=False, **kwargs):
        super(QValuePlotter, self).__init__(agent)

        self.q_value_subplotter = QValueSubplotter(agent, corl_colors,
                                                   write_values, **kwargs)
        if safety_truth is not None:
            self.safety_truth_subplotter = SafetyTruthSubplotter(safety_truth,
                                                                 corl_colors)
        else:
            self.safety_truth_subplotter = None
        self.plot_samples = plot_samples
        if self.plot_samples:
            self.sample_subplotter = SampleSubplotter(corl_colors)

    def get_figure(self):
        figure = plt.figure(constrained_layout=True, figsize=(4, 4))

        ax_Q = figure.add_subplot()
        ax_Q.tick_params(direction='in', top=True, right=True)

        Q_values = self.get_subplotters_params()

        if self.safety_truth_subplotter is not None:
            self.safety_truth_subplotter.draw_on_axs(ax_Q)
        q_values_image = self.q_value_subplotter.draw_on_axs(ax_Q, Q_values)
        if self.plot_samples:
            self.sample_subplotter.draw_on_axs(ax_Q)

        figure.colorbar(q_values_image, ax=ax_Q, location='right')

        plt.title('Q-Values map (interpolated)')
        return figure

    def get_subplotters_params(self):
        Q_values = self.agent.Q_model[:, :].reshape(
            self.agent.env.stateaction_space.shape
        )
        return Q_values

    def on_run_iteration(self, state, action, new_state, reward, failed, *args,
                         **kwargs):
        if self.plot_samples:
            self.sample_subplotter.incur_sample(state, action, failed)


class DiscreteQValuePlotter(QValuePlotter):
    def __init__(self, agent, safety_truth=None, write_values=False,
                 plot_samples=False, vmin=None, vmax=None):
        super(DiscreteQValuePlotter, self).__init__(agent, safety_truth, write_values,
                                            plot_samples, vmin=vmin, vmax=vmax)
        self.q_value_subplotter = DiscreteQValueSubplotter(agent, corl_colors,
                                                   write_values, vmin, vmax)
        self.vmin = vmin
        self.vmax = vmax

    def get_subplotters_params(self):
        Q_values = self.agent.Q_model[:, :].reshape(
            self.agent.env.stateaction_space.shape
        )
        if isinstance(self.agent.env, DiscreteHovership):
            vmin = Q_values[1, :].min()
            Q_values[0,:] = vmin
            print(f'WARNING: Q_values[0,:] were set to {vmin} for plotting.')
        return Q_values