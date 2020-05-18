from pathlib import Path
import os


class Simulation:
    def __init__(self, output_directory, name, plotters):
        self.output_directory = Path(output_directory) / name
        self.name = name
        self.plotters = plotters

        self.fig_path = self.output_directory / 'figs'
        self.log_path = self.output_directory / 'logs'

        self.fig_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=False, exist_ok=True)

        self.__saved_figures = {}

    def run(self):
        raise NotImplementedError

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
