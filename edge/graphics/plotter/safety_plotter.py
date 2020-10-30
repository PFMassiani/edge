import matplotlib.pyplot as plt


from . import Plotter
from ..subplotter import SafetyMeasureSubplotter, SafetyTruthSubplotter,\
    SampleSubplotter, SafetyGPSubplotter
from ..colors import corl_colors


class SafetyPlotter(Plotter):
    def __init__(self, agent, ground_truth=None):
        super(SafetyPlotter, self).__init__(agent)

        self.safety_subplotter = SafetyMeasureSubplotter(agent, corl_colors)
        self.sample_subplotter = SampleSubplotter(corl_colors)
        if ground_truth is not None:
            self.truth_subplotter = SafetyTruthSubplotter(ground_truth,
                                                          corl_colors)
        else:
            self.truth_subplotter = None

    def get_figure(self):
        figure = plt.figure(constrained_layout=True, figsize=(5.5, 4.8))
        gs = figure.add_gridspec(1, 2, width_ratios=[3, 1])

        ax_Q = figure.add_subplot(gs[0, 0])
        ax_S = figure.add_subplot(gs[0, 1], sharey=ax_Q)

        ax_Q.tick_params(direction='in', top=True, right=True)
        ax_S.tick_params(direction='in', left=False)

        if self.truth_subplotter is not None:
            self.truth_subplotter.draw_on_axs(ax_Q, ax_S)
        Q_optimistic, Q_cautious, S_optimistic = self.get_subplotters_params()
        self.safety_subplotter.draw_on_axs(ax_Q, ax_S, Q_optimistic,
                                           Q_cautious, S_optimistic)
        self.sample_subplotter.draw_on_axs(ax_Q)

        plt.title('Safety measure')
        return figure

    def get_subplotters_params(self):
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
        S_optimistic = Q_optimistic.mean(axis=1)
        return Q_optimistic, Q_cautious, S_optimistic

    def on_run_iteration(self, state, action, new_state, reward, failed,
                         color=None):
        self.sample_subplotter.incur_sample(state, action, failed, color)


class DetailedSafetyPlotter(Plotter):
    def __init__(self, agent, ground_truth=None):
        super(DetailedSafetyPlotter, self).__init__(agent)

        self.safety_subplotter = SafetyMeasureSubplotter(agent, corl_colors)
        self.safety_gpinfo_subplotter = SafetyGPSubplotter(agent, corl_colors)
        self.sample_subplotter = SampleSubplotter(corl_colors)
        if ground_truth is not None:
            self.truth_subplotter = SafetyTruthSubplotter(ground_truth,
                                                          corl_colors)
        else:
            self.truth_subplotter = None

    def get_figure(self):
        figure = plt.figure(constrained_layout=True, figsize=(5.5, 4.8))
        gs = figure.add_gridspec(3, 2, width_ratios=[3, 1])

        ax_Q = figure.add_subplot(gs[0, 0])
        ax_S = figure.add_subplot(gs[0, 1], sharey=ax_Q)
        ax_Meas = figure.add_subplot(gs[1,0], sharex=ax_Q)
        ax_Meas_cbar = figure.add_subplot(gs[1,1])
        ax_Var = figure.add_subplot(gs[2,0], sharex=ax_Q)
        ax_Var_cbar = figure.add_subplot(gs[2,1])

        ax_Q.tick_params(direction='in', top=True, right=True)
        ax_S.tick_params(direction='in', left=False)
        ax_Meas.tick_params(direction='in', top=True, right=True)
        ax_Var.tick_params(direction='in', top=True, right=True)

        if self.truth_subplotter is not None:
            self.truth_subplotter.draw_on_axs(ax_Q, ax_S)
        Q_optimistic, Q_cautious, measure, covar, S_optimistic = \
            self.get_subplotters_params()
        self.safety_subplotter.draw_on_axs(ax_Q, ax_S, Q_optimistic,
                                           Q_cautious, S_optimistic)
        meas_image, var_image= self.safety_gpinfo_subplotter.draw_on_axs(
            ax_Meas, ax_Var, measure, covar
        )
        self.sample_subplotter.draw_on_axs(ax_Q)
        self.sample_subplotter.draw_on_axs(ax_Meas)
        self.sample_subplotter.draw_on_axs(ax_Var)

        ax_Meas_cbar.axis('off')
        ax_Var_cbar.axis('off')

        figure.colorbar(meas_image, ax=ax_Meas_cbar, location='left')
        figure.colorbar(var_image, ax=ax_Var_cbar, location='left')

        return figure

    def get_subplotters_params(self):
        Q_list, covar, measure = self.safety_subplotter.model.level_set(
            state=None,
            lambda_threshold=[
                0,
                self.agent.lambda_cautious
            ],
            gamma_threshold=[
                self.agent.gamma_optimistic,
                self.agent.gamma_cautious
            ],
            return_proba=False,
            return_covar=True,
            return_measure=True
        )
        Q_optimistic, Q_cautious = Q_list

        Q_optimistic = Q_optimistic.reshape(
            self.agent.env.stateaction_space.shape
        ).astype(float)
        Q_cautious = Q_cautious.reshape(
            self.agent.env.stateaction_space.shape
        ).astype(float)
        measure = measure.reshape(
            self.agent.env.stateaction_space.shape
        )
        covar = covar.reshape(
            self.agent.env.stateaction_space.shape
        )
        # The action axis is 1 because we can only plot 2D Stateaction spaces
        S_optimistic = Q_optimistic.mean(axis=1)
        return Q_optimistic, Q_cautious, measure, covar, S_optimistic

    def on_run_iteration(self, state, action, new_state, reward, failed):
        self.sample_subplotter.incur_sample(state, action, failed)