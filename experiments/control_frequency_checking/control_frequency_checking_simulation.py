from pathlib import Path
import logging
import numpy as np

from edge import Simulation
from edge.envs.continuous_cartpole import ContinuousCartPole

from control_frequency_checking_parameterization import DLQRController


class ControlFrequencyCheckingSimulation(Simulation):
    def __init__(self, name, env_name, Q, R, discretization_shape,
                 control_frequencies, render=False):
        if env_name == 'cartpole':
            env_builder = ContinuousCartPole
        else:
            raise ValueError(f'Unsupported environment {env_name}')

        output_directory = Path(__file__).parent.resolve()
        super(ControlFrequencyCheckingSimulation, self).__init__(
            output_directory, name, {}
        )

        self.Q = Q
        self.R = R
        self.control_frequencies = []
        self.agents = []
        for cf in control_frequencies:
            try:
                agent = DLQRController(
                    env_builder(discretization_shape=discretization_shape,
                                control_frequency=cf),
                    Q=Q,
                    R=R,
                )
                self.agents.append(agent)
                self.control_frequencies.append(cf)
            except np.linalg.LinAlgError:
                logging.warning(f'Failed to solve the ARE for control frequency'
                                f' {cf}')
        self.render = render

    def run_agent(self, agent):
        n_episodes = 0
        max_episodes = 10
        failures = 0
        successes = 0
        while n_episodes < max_episodes:
            done = False
            failed = True
            n_steps = 0
            r = 0
            agent.reset()
            while not done:
                new_state, reward, failed, done = agent.step()
                r += reward
                if self.render:
                    agent.env.render()
            if failed:
                failures += 1
            else:
                successes += 1
            n_episodes += 1
        return successes, failures

    def run(self):
        logging.info(f'Simulation started with \nQ={self.Q}\nR={self.R}')
        for cf, agent in zip(self.control_frequencies, self.agents):
            successes, failures = self.run_agent(agent)
            logging.info(f'Control frequency: {cf} | {successes} successes | '
                         f'{failures} failures')


if __name__ == '__main__':
    Q = np.eye(4)
    R = 0.4
    control_frequencies = list(range(1, 11))
    sim = ControlFrequencyCheckingSimulation(
        env_name='cartpole',
        name='cartpole',
        Q=Q,
        R=R,
        discretization_shape=(10, 10, 10, 10, 10),
        control_frequencies=control_frequencies,
        render=True,
    )
    sim.run()
