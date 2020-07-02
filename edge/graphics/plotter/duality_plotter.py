from matplotlib import pyplot as plt
from numpy import linspace

from . import Plotter
from ..subplotter import ValueSubplotter
from ..colors import corl20_colors


class DualityPlotter(Plotter):
    def __init__(self, penalties, agent_con, *agents_pen):
        super(DualityPlotter, self).__init__(None)

        self.value_con_subplotter = ValueSubplotter(agent_con, corl20_colors, constrained=True)
        self.value_pen_t = linspace(0, 1, len(agents_pen))
        self.value_pen_subplotter = [ValueSubplotter(a_pen, corl20_colors, constrained=False)
                                     for a_pen in agents_pen]
        self.label_con = 'constrained'
        self.label_pen = [f'p={p}' for p in penalties]

    def get_figure(self):
        figure = plt.figure(constrained_layout=True,figsize=(4, 4))

        ax_V = figure.add_subplot()
        ax_V.tick_params(direction='in', top=True, right=True)

        def call_subplotter(subplotter, label, t=None):
            Q_values = self.get_subplotter_params(subplotter)
            subplotter.draw_on_axs(ax_V, Q_values, label, t)

        call_subplotter(self.value_con_subplotter, self.label_con)
        for v_pen_subplotter, t, lbl in zip(self.value_pen_subplotter, self.value_pen_t, self.label_pen):
            call_subplotter(v_pen_subplotter, lbl, t)

        plt.yscale('symlog')
        # plt.grid(True)
        plt.title('Values map')
        return figure


    def get_subplotter_params(self, subplotter):
        Q_values = subplotter.model[:, :].reshape(
            subplotter.agent.env.stateaction_space.shape
        )
        return Q_values


    def on_run_iteration(self, state, action, new_state, reward, failed):
        pass