from matplotlib import pyplot as plt
from numpy import linspace, inf

from . import Plotter
from ..subplotter import ValueSubplotter
from ..colors import corl20_colors
from edge.envs import DiscreteHovership


class DualityPlotter(Plotter):
    def __init__(self, penalties, agent_con, *agents_pen):
        super(DualityPlotter, self).__init__(None)

        self.value_con_subplotter = ValueSubplotter(agent_con, corl20_colors, constrained=True)
        self.value_pen_t = linspace(0, 1, len(agents_pen))
        self.value_pen_subplotter = [ValueSubplotter(a_pen, corl20_colors, constrained=False)
                                     for a_pen in agents_pen]
        self.label_con = r'Constrained'
        self.label_pen = [r'$p=' + str(p) + r'$' for p in penalties]

    def get_figure(self):
        figure = plt.figure(constrained_layout=True,figsize=(4, 4))

        ax_V = figure.add_subplot()
        ax_V.tick_params(direction='in', top=True, right=True)

        def call_subplotter(subplotter, label, t=None, fill_from=None):
            Q_values = self.get_subplotter_params(subplotter)
            subplotter.draw_on_axs(ax_V, Q_values, label, t, fill_from)
            return Q_values.min()

        v_min = inf
        for v_pen_subplotter, t, lbl in zip(self.value_pen_subplotter, self.value_pen_t, self.label_pen):
            running_v_min = call_subplotter(v_pen_subplotter, lbl, t)
            v_min = min(v_min, running_v_min)

        call_subplotter(self.value_con_subplotter, self.label_con,
                        fill_from=v_min)

        plt.yscale('symlog')
        plt.grid(True)
        plt.title(r'Values map')
        return figure


    def get_subplotter_params(self, subplotter):
        Q_values = subplotter.model[:, :].reshape(
            subplotter.agent.env.stateaction_space.shape
        )
        if isinstance(subplotter.agent.env, DiscreteHovership):
            vmin = Q_values[1, :].min()
            Q_values[0,:] = vmin
            print(f'WARNING: Q_values[0,:] were set to {vmin} for plotting.')
        return Q_values


    def on_run_iteration(self, state, action, new_state, reward, failed):
        pass