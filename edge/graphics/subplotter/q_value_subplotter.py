from numpy import squeeze

from . import Subplotter


class QValueSubplotter(Subplotter):
    def __init__(self, agent, colors):
        super(QValueSubplotter, self).__init__(colors)
        self.agent = agent
        self.states = squeeze(self.model.env.state_space[:])
        self.actions = squeeze(self.model.env.action_space[:])
        stateaction_grid = self.model.env.stateaction_space[:, :]
        self.states_grid = stateaction_grid[:, :, 0]
        self.actions_grid = stateaction_grid[:, :, 1]

        self.min = None
        self.max = None

    @property
    def model(self):
        return self.agent.Q_model

    def draw_on_axs(self, ax_Q, Q_values):
        qmin = Q_values.min()
        qmax = Q_values.max()
        if self.min is None or qmin < self.min:
            self.min = qmin
        if self.max is None or qmax > self.max:
            self.max = qmax
        image = ax_Q.pcolormesh(
            self.actions_grid,
            self.states_grid,
            Q_values,
            cmap=self.colors.cmap_q_values,
            vmin=-10,  # self.min,
            vmax=self.max,
            alpha=0.7
        )

        ax_Q.set_xlabel('action space $A$')
        ax_Q.set_ylabel('state space $S$')
        frame_width_x = self.actions[-1] * .03
        ax_Q.set_xlim((self.actions[0] - frame_width_x,
                       self.actions[-1] + frame_width_x))

        frame_width_y = self.states[-1] * .03
        ax_Q.set_ylim((self.states[0] - frame_width_y,
                       self.states[-1] + frame_width_y))

        return image
