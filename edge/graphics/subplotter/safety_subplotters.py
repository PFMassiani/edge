from numpy import squeeze
import matplotlib
from numpy import abs as npabs

from . import Subplotter


class SafetyMeasureSubplotter(Subplotter):
    def __init__(self, agent, colors):
        super(SafetyMeasureSubplotter, self).__init__(colors)
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

    @property
    def model(self):
        return self.agent.safety_model

    def draw_on_axs(self, ax_Q, ax_S, Q_optimistic, Q_cautious, S_optimistic):
        self.draw_on_Q(ax_Q, Q_optimistic, Q_cautious)
        self.draw_on_S(ax_S, S_optimistic)

    def draw_on_Q(self, ax_Q, Q_optimistic, Q_cautious):
        def draw_contour(Q):
            ax_Q.contour(
                self.actions_grid,
                self.states_grid,
                Q,
                [.5],
                colors='k'
            )

        def fill_contour(Q, color):
            ax_Q.contourf(
                self.actions_grid,
                self.states_grid,
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


class SafetyTruthSubplotter(Subplotter):
    def __init__(self, ground_truth, colors):
        self.ground_truth = ground_truth
        self.colors = colors
        self.states = squeeze(
            self.ground_truth.stateaction_space.state_space[:]
        )
        self.actions = squeeze(
            self.ground_truth.stateaction_space.action_space[:]
        )

        stateaction_grid = self.ground_truth.stateaction_space[:, :]
        self.states_grid = stateaction_grid[:, :, 0]
        self.actions_grid = stateaction_grid[:, :, 1]

    def draw_on_axs(self, ax_Q, ax_S):
        self.draw_on_Q(ax_Q)
        self.draw_on_S(ax_S)

    def draw_on_Q(self, ax_Q):
        def draw_contour(Q):
            ax_Q.contour(
                self.actions_grid,
                self.states_grid,
                Q,
                [.5],
                colors='k'
            )

        def fill_contour(Q, color, hatch):
            matplotlib.rcParams['hatch.color'] = color
            ax_Q.contourf(
                self.actions_grid,
                self.states_grid,
                Q,
                [.5, 2],
                colors='w',
                hatches=[hatch, None]
            )
        draw_contour(self.ground_truth.viable_set)
        fill_contour(self.ground_truth.viable_set, self.colors.truth, '--')
        fill_contour(self.ground_truth.failure_set, self.colors.failure, 'XX')
        fill_contour(self.ground_truth.unviable_set, self.colors.unviable,
                     '//')

    def draw_on_S(self, ax_S):
        ax_S.plot(self.ground_truth.state_measure, self.states, color='k')

        ax_S.fill_betweenx(self.states, 0, self.ground_truth.state_measure,
                           hatch='--',
                           facecolor='none', edgecolor=self.colors.truth)

        ax_S.set_xlim((0, max(self.ground_truth.state_measure) * 1.2))
        ax_S.get_yaxis().set_visible(False)
        ax_S.set_xlabel('$\Lambda$')


class SafetyGPSubplotter(Subplotter):
    def __init__(self, agent, colors):
        super(SafetyGPSubplotter, self).__init__(colors)
        self.agent = agent
        stateaction_grid = self.agent.safety_model.env.stateaction_space[:, :]
        self.states_grid = stateaction_grid[:, :, 0]
        self.actions_grid = stateaction_grid[:, :, 1]

        self.max_var = None
        self.max_meas = None

    def draw_on_axs(self, ax_Meas, ax_Var, meas, covar):
        meas_image = self.draw_meas(ax_Meas, meas)
        var_image = self.draw_var(ax_Var, covar)
        return meas_image, var_image

    def draw_var(self, ax_Var, covar):
        varmax = covar.max()
        if self.max_var is None or varmax > self.max_var:
            self.max_var = varmax
        return ax_Var.pcolormesh(
            self.actions_grid,
            self.states_grid,
            covar,
            cmap=self.colors.cmap_var,
            vmin=0,
            vmax=self.max_var
        )

    def draw_meas(self, ax_Meas, meas):
        measmax = npabs(meas).max()
        if self.max_meas is None or measmax > self.max_meas:
            self.max_meas = measmax
        return ax_Meas.pcolormesh(
            self.actions_grid,
            self.states_grid,
            meas,
            cmap=self.colors.cmap_meas,
            vmin=-self.max_meas,
            vmax=self.max_meas
        )