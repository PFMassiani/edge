import argparse
from pathlib import Path
import logging
from edge.utils.logging import config_msg
from benchmark_simulation import BenchmarkSingleSimulation
from benchmark_parameterizations import SIMULATION_PARAMS, APARAMS_DICT, \
    ENVPARAMS_DICT

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('rid')
    parser.add_argument('environment')
    parser.add_argument('agent')

    args = parser.parse_args()

    output_directory = Path('.') / args.environment / args.agent
    name = str(args.rid)

    simulation_parameters = SIMULATION_PARAMS[args.agent].copy()
    simulation_parameters.update({
        'output_directory': output_directory.absolute(),
        'name': name,
        'envname': args.environment,
        'aname': args.agent,
        'envparams': ENVPARAMS_DICT[args.environment],
        'aparams': APARAMS_DICT[args.agent][args.environment],
    })

    sim = BenchmarkSingleSimulation(seed=int(args.rid), **simulation_parameters)
    logger.info(config_msg(f'Benchmark script called with arguments: {args}'))
    logger.info(config_msg('Simulation created with parameterization:'))
    logger.info(config_msg(f'{simulation_parameters}'))
    logger.info(f'Starting simulation {name}')
    sim.run()
    logger.info(f'Simulation completed. Outputs were saved in '
                f'{output_directory.absolute()}')
