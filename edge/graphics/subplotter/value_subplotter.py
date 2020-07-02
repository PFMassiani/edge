import warnings
from numpy import squeeze
import matplotlib as mpl

from . import Subplotter

class ValueSubplotter(Subplotter):
    def __init__(self, agent, colors, constrained=True):
        super(ValueSubplotter, self).__init__(colors)
        self.agent = agent
        self.states = squeeze(self.model.env.state_space[:])
        if self.states.ndim != 1:
            raise ValueError(f'Expected a 1-dimensional state space, got {self.states.ndim} dimensions instead')
        self.actions_axes = tuple([1 + k
                              for k in range(self.agent.env.action_space.index_dim)])
        self.constrained = constrained
        if constrained:
            self.plot_filter = self.agent.constraint.viable_set.any(axis=self.actions_axes)
            self.states = self.states[self.plot_filter]

    @property
    def model(self):
        return self.agent.Q_model

    def draw_on_axs(self, ax_V, Q_values, label, t=None):
        warnings.filterwarnings('ignore')
        values = Q_values.max(axis=self.actions_axes)

        if self.constrained:
            color = self.colors.value_con
            plot_values = values[self.plot_filter]
            linewidth = self.colors.value_con_width
        else:
            color = self.colors.value_pen_cm(t)
            plot_values = values
            linewidth = self.colors.value_pen_width
        ax_V.plot(
            self.states, plot_values,
            color=color,
            linewidth=linewidth,
            label=label
        )

        ax_V.set_xticks(self.states)

        ax_V.set_xlabel('state')
        ax_V.set_ylabel('value')

        ax_V.legend()

        return ax_V