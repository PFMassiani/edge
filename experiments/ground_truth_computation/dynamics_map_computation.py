from pathlib import Path
import logging
logger = logging.getLogger(__name__)
import time
from numpy import save as npsave

from edge.envs import ContinuousCartPole
from edge.utils.logging import config_msg

from ground_truth_computation_parameterization import \
    TruthComputationSimulation, Q_map_name


class DynamicsMapComputation(TruthComputationSimulation):
    def __init__(self, name, env_name, discretization_shape, *args, **kwargs):
        if env_name == 'cartpole':
            env_builder = ContinuousCartPole
        else:
            raise ValueError(f'Environment {env_name} is not supported')
        self.env = env_builder(discretization_shape=discretization_shape,
                          *args, **kwargs)

        output_directory = Path(__file__).parent.resolve()
        super(DynamicsMapComputation, self).__init__(
            output_directory, name, Q_map_name(env_name)
        )
        self.save_path = self.output_directory / Q_map_name(env_name)

        self.Q_map = None
        logger.info(config_msg(f"env_name='{env_name}'"))
        logger.info(
            config_msg(f"discretization_shape='{discretization_shape}'")
        )
        logger.info((config_msg(f"args={args}")))
        logger.info((config_msg(f"kwargs={kwargs}")))

    def run(self):
        logger.info('Launched computation of dynamics map')
        tick = time.time()
        self.Q_map = self.env.compute_dynamics_map()
        tock = time.time()
        logger.info(f'Done in {tock-tick:.2f} s.')
        npsave(str(self.save_path), self.Q_map, allow_pickle=True)
        logger.info(f'Output saved in {str(self.save_path)}.npy')


if __name__ == '__main__':
    sim = DynamicsMapComputation(
        name='cartpole_4',
        env_name='cartpole',
        discretization_shape=(50, 50, 50, 50, 50),
        control_frequency=4,
    )
    sim.run()