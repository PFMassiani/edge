import logging
from pathlib import Path

from edge.utils.logging import ConfigFilter


def Q_map_name(env_name):
    return f'{env_name}_Q_map'


def safety_name(env_name):
    return f'{env_name}_safety'


class TruthComputationSimulation:
    def __init__(self, output_directory, name, log_name):
        self.name = name
        self.output_directory = Path(output_directory) / name

        self.log_path = self.output_directory / 'logs'
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.log_name = log_name

        self.setup_default_logging_configuration()

    def setup_default_logging_configuration(self):
        training_handler = logging.FileHandler(
            self.log_path / f'{self.log_name}.log'
        )
        training_handler.addFilter(ConfigFilter(log_if_match=False))
        training_handler.setLevel(logging.INFO)
        config_handler = logging.FileHandler(
            self.log_path / f'{self.log_name}.conf'
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
