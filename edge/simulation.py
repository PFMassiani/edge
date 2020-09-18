from pathlib import Path
import os
import logging
from matplotlib.pyplot import close as plt_close
from numpy.random import seed as npseed

from edge.utils.logging import ConfigFilter


class Simulation:
    """
    Base class for a Simulation. Takes care of defining the agent, the main loop, and saving the results and figures in
    the appropriate locations.
    """
    def __init__(self, output_directory, name, plotters):
        self.set_seed()
        self.output_directory = Path(output_directory) / name
        self.name = name
        self.plotters = plotters if plotters is not None else {}

        self.fig_path = self.output_directory / 'figs'
        self.log_path = self.output_directory / 'logs'

        self.fig_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=False, exist_ok=True)

        self.__saved_figures = {}

        self.setup_default_logging_configuration()

    def set_seed(self, value=None):
        npseed(value)

    def run(self):
        raise NotImplementedError

    def on_run_iteration(self, *args, **kwargs):
        for plotter in self.plotters.values():
            try:
                plotter.on_run_iteration(*args, **kwargs)
            except AttributeError as e:
                # The plotter does not have a on_run_iteration routine:
                #  this is not a problem.
                pass

    def save_figs(self, prefix):
        for name, plotter in self.plotters.items():
            savename = prefix + '_' + name + '.pdf'
            savepath = self.fig_path / savename
            fig = plotter.get_figure()
            fig.savefig(str(savepath), format='pdf')

            if self.__saved_figures.get(name) is None:
                self.__saved_figures[name] = [str(savepath)]
            else:
                self.__saved_figures[name] += [str(savepath)]

            plt_close('all')

    def compile_gif(self):
        for name, figures in self.__saved_figures.items():
            figures_to_compile = ' '.join(figures)
            path = str(self.fig_path)
            gif_command = ("convert -delay 50 -loop 0 -density 300 "
                           f"{figures_to_compile} {path}/{name}.gif")

            try:
                os.system(gif_command)
            except Exception as e:
                print(f'Error: could not compile {name}.gif. Exception: {e}')

    def setup_default_logging_configuration(self):
        training_handler = logging.FileHandler(
            self.log_path / 'training.log'
        )
        training_handler.addFilter(ConfigFilter(log_if_match=False))
        training_handler.setLevel(logging.INFO)
        config_handler = logging.FileHandler(
            self.log_path / 'config.log'
        )
        config_handler.addFilter(ConfigFilter(log_if_match=True))
        config_handler.setLevel(logging.INFO)
        stdout_handler = logging.StreamHandler()
        stdout_handler.setLevel(logging.INFO)

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(training_handler)
        root_logger.addHandler(config_handler)
        root_logger.addHandler(stdout_handler)


class ModelLearningSimulation(Simulation):
    """
    Adds the notion of Model to the base Simulation, and enables saving these models.
    """
    def __init__(self, output_directory, name, plotters):
        super(ModelLearningSimulation, self).__init__(
            output_directory, name, plotters
        )
        self.models_path = Path(__file__).parent / 'data' / 'models'
        self.local_models_path = self.output_directory / 'models'
        self.local_models_path.mkdir(exist_ok=True)
        self.samples_path = self.output_directory / 'samples'
        self.samples_path.mkdir(exist_ok=True)

    def get_models_to_save(self):
        raise NotImplementedError

    def save_models(self, globally=False, locally=True):
        models_to_save = self.get_models_to_save()
        paths_where_to_save = []
        if globally:
            paths_where_to_save.append(self.models_path)
        if locally:
            paths_where_to_save.append(self.local_models_path)

        for path in paths_where_to_save:
            for savename, model in models_to_save.items():
                savepath = path / savename
                savepath.mkdir(exist_ok=True)
                model.save(savepath)

    def load_models(self, skip_local=False):
        raise NotImplementedError

    def save_samples(self, name):
        models_to_save = self.get_models_to_save()
        for savename, model in models_to_save.items():
            model.save_samples(self.samples_path / savename / name)

    def load_samples(self, name):
        models_to_load = self.get_models_to_save()
        for savename, model in models_to_load.items():
            model.load_samples(self.samples_path / savename / name)
