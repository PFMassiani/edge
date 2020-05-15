from . import Subplotter


class SafetyMeasureSubplotter(Subplotter):
    def __init__(self, agent, model, colors):
        super(SafetyMeasureSubplotter, self).__init__(model, colors)
        self.agent = agent
        self.states = self.model.env.state_space[:]
        self.actions = self.model.env.action_space[:]

        if not len(self.states.shape) == 1:
            raise ValueError('Too many state dimensions for plotting')
        if not len(self.actions.shape) == 1:
            raise ValueError('Too many action dimensions for plotting')

        self.nS = self.states.shape[0]
        self.nA = self.actions.shape[0]

        stateaction_grid = self.model.env.stateaction_space[:, :]
        self.states_grid = stateaction_grid[:, :, 0]
        self.actions_grid = stateaction_grid[:, :, 1]

    def draw_on_axs(self, ax_Q, ax_S):
        self.Q_optimistic, self.Q_cautious = self.model.full_level_set(
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

        self.draw_on_Q(ax_Q)
        self.draw_on_S(ax_S)

    def draw_on_Q(self, ax_Q):
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

        draw_contour(self.Q_optimistic)
        draw_contour(self.Q_cautious)
        fill_contour(self.Q_optimistic, self.colors.optimistic)
        fill_contour(self.Q_cautious, self.colors.cautious)

    def draw_on_S(self, ax_S):
        # TODO
        pass
