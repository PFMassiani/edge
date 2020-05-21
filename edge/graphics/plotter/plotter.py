import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, agent):
        self.agent = agent

    def get_figure(self):
        raise NotImplementedError

    def on_run_iteration(self):
        raise NotImplementedError
