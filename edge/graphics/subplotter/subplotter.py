class Subplotter:
    def __init__(self, colors):
        self.colors = colors

    def draw_on_axs(self):
        raise NotImplementedError

    def on_run_iteration(self, state, action, new_state, reward, failed):
        pass