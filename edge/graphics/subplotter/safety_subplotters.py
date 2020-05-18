from numpy import squeeze

from . import Subplotter


class SafetyMeasureSubplotter(Subplotter):
    def __init__(self, agent, model, colors):
        super(SafetyMeasureSubplotter, self).__init__(model, colors)
        self.agent = agent
        self.states = squeeze(self.model.env.state_space[:])
        self.actions = squeeze(self.model.env.action_space[:])

        if not len(self.states.shape) == 1:
            raise ValueError('Too many state dimensions for plotting: '
                             'expected 1 dimension, got '
                             f'{len(self.states.shape)}')
        if not len(self.actions.shape) == 1:
            raise ValueError('Too many action dimensions for plotting: '
                             'expected 1 dimension, got '
                             f'{len(self.actions.shape)}')

        self.nS = self.states.shape[0]
        self.nA = self.actions.shape[0]

        stateaction_grid = self.model.env.stateaction_space[:, :]
        self.states_grid = stateaction_grid[:, :, 0]
        self.actions_grid = stateaction_grid[:, :, 1]

    def draw_on_axs(self, ax_Q, ax_S, Q_optimistic, Q_cautious, S_optimistic):
        self.draw_on_Q(ax_Q, Q_optimistic, Q_cautious)
        self.draw_on_S(ax_S, S_optimistic)

    def draw_on_Q(self, ax_Q, Q_optimistic, Q_cautious):
        def draw_contour(Q):
            ax_Q.contour(
                self.states_grid,
                self.actions_grid,
                Q,
                [.5],
                colors='k'
            )

        def fill_contour(Q, color):
            ax_Q.contourf(
                self.states_grid,
                self.actions_grid,
                Q,
                [.5, 2],
                colors=[color, (0, 0, 0, 0)],
                alpha=0.7
            )
        draw_contour(Q_optimistic)
        draw_contour(Q_cautious)
        fill_contour(Q_optimistic, self.colors.optimistic)
        fill_contour(Q_cautious, self.colors.cautious)

        ax_Q.set_xlabel('action space $A$')
        ax_Q.set_ylabel('state space $S$')
        frame_width_x = self.actions[-1] * .03
        ax_Q.set_xlim((self.actions[0] - frame_width_x,
                       self.actions[-1] + frame_width_x))

        frame_width_y = self.states[-1] * .03
        ax_Q.set_ylim((self.states[0] - frame_width_y,
                       self.states[-1] + frame_width_y))

    def draw_on_S(self, ax_S, S_optimistic):
        ax_S.plot(S_optimistic, self.states, color='k')
        ax_S.fill_betweenx(self.states, 0, S_optimistic,
                           facecolor=self.colors.optimistic, alpha=0.7)

        ax_S.set_xlim((0, max(S_optimistic) * 1.2))
        ax_S.get_yaxis().set_visible(False)
        ax_S.set_xlabel('$\Lambda$')
