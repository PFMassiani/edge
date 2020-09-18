from pathlib import Path
import logging

logger = logging.getLogger(__name__)
import time
from numpy import savez

from edge.envs import ContinuousCartPole
from edge.model.safety_models import SafetyTruth
from edge.utils.logging import config_msg

from ground_truth_computation_parameterization import \
    TruthComputationSimulation, Q_map_name, safety_name


class SafetyTruthComputation(TruthComputationSimulation):
    def __init__(self, name, env_name, discretization_shape, *args, **kwargs):
        if env_name == 'cartpole':
            env_builder = ContinuousCartPole
        else:
            raise ValueError(f'Environment {env_name} is not supported')
        output_directory = Path(__file__).parent.resolve()
        super(SafetyTruthComputation, self).__init__(
            output_directory, name, safety_name(env_name)
        )

        self.env = env_builder(discretization_shape=discretization_shape,
                               *args, **kwargs)
        self.truth = SafetyTruth(self.env)

        self.Q_map_path = self.output_directory / (str(Q_map_name(env_name)) +
                                                   '.npy')
        self.save_path = self.output_directory / safety_name(env_name)

        logger.info(config_msg(f"env_name='{env_name}'"))
        logger.info(
            config_msg(f"discretization_shape='{discretization_shape}'")
        )
        logger.info((config_msg(f"args={args}")))
        logger.info((config_msg(f"kwargs={kwargs}")))

    def run(self):
        logger.info('Launched computation of viable set')
        if not self.Q_map_path.exists():
            errormsg = f'The transition map could not be found at ' \
                       f'{str(self.Q_map_path)}. Please compute it first.'
            logger.critical(errormsg)
            raise FileNotFoundError(errormsg)
        tick = time.time()
        self.truth.compute(self.Q_map_path)
        tock = time.time()
        logger.info(f'Done in {tock - tick:.2f} s.')
        self.truth.save(str(self.save_path))
        logger.info(f'Output saved in {str(self.save_path)}')


if __name__ == '__main__':
    sim = SafetyTruthComputation(
        name='cartpole_test',
        env_name='cartpole',
        discretization_shape=(2, 2, 2, 2, 2),
        control_frequency=4,
    )
    sim.run()
