class SubPlotter:
    def __init__(self, model, colors):
        self.model = model
        self.colors = colors

    def draw_on_axs(self):
        raise NotImplementedError
