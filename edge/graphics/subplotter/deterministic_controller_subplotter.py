import numpy as np

from . import Subplotter


class DeterministicControllerSubplotter(Subplotter):
    def __init__(self, stateaction_space, controller, colors):
        super().__init__(colors)
        self.stateaction_space = stateaction_space
        self.controller = controller

        if self.stateaction_space.data_length > 2:
            raise ValueError("Can only plot controllers for 2D state-actions")

    def draw_on_axs(self, ax_Q):
        states = [s for _, s in iter(self.stateaction_space.state_space)]
        actions = [self.controller.get_action(s) for s in states]

        states = np.array(states).flatten()
        actions = np.array(actions).flatten()

        ax_Q.plot(
            actions,
            states,
            color=[1, 0, 0],
            linewidth=1.5
        )
