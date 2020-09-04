import argparse
from pathlib import Path
import numpy as np
from benchmark_simulation import BenchmarkSingleSimulation as b


PENALTY_KEY = 'penalty_level'
REWARDS = 'rewards'
FAILURES = 'failed'

def ls(p):
	return p.iterdir()

def get_name(p):
	return str(p.absolute()).split('/')[-1]

def extract_penalty_from_conf_line(line):
	# Largely relies on the structure of the log!
	dict_as_str = line.split('[CONFIG] ')[1]
	penalty_entry = dict_as_str.split(',')[0]
	penalty = penalty_entry.split(':')[1]
	return int(penalty)

def resort_on_meas_number(meas_list):
	meas_array = np.array(meas_list)
	sorted_idx = np.argsort(meas_array[:, 0])
	return meas_array[sorted_idx]

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('env')

	args = parser.parse_args()
	envname = 'penalized_' + args.env

	base_path = Path('./results/') / envname

	for apath in ls(base_path):
		dirname = get_name(apath)
		if dirname == 'metrics':
			continue # Go to next iteration
		for exp_path in ls(apath):
			conf_path = exp_path / 'logs' / 'config.log'
			with conf_path.open('r') as f:
				for line in f:
					if PENALTY_KEY in line:
						penalty = extract_penalty_from_conf_line(line)
						break

			samples_path = exp_path / 'samples'
			corrected_exp_reward = []
			corrected_std_reward = []
			for measurement_episode_file in ls(samples_path):
				measurement_name = get_name(measurement_episode_file)[:-4]  # Remove npz extension
				if measurement_name == 'training':
					continue
				masurement_number = int(measurement_name.split('_')[-1])
				meas = np.load(measurement_episode_file)
				n_episodes = 20
				measurements = [{REWARDS: [], FAILURES: []} for ep in range(n_episodes)]
				for key in meas.keys():
					mname, ep = key.split('_EPISODE_')
					if mname in [REWARDS, FAILURES]:
						ep = int(ep)
						measurements[ep][mname] = meas[key]
				corrected_measurement_rewards = [None] * n_episodes
				for ep in range(n_episodes):
					if measurements[ep][FAILURES][-1]:
						measurements[ep][REWARDS][-1] += penalty  # Remove the additive penalty
					corrected_measurement_rewards[ep] = sum(measurements[ep][REWARDS])
				corrected_exp_reward.append(
					[measurement_number, np.mean(corrected_measurement_rewards)]
				)
				corrected_std_reward.append(
					[measurement_number, np.std(corrected_measurement_rewards)]
				)
			corrected_exp_reward = resort_on_meas_number(corrected_exp_reward)
			corrected_std_reward = resort_on_meas_number(corrected_std_reward)

			try:
				metrics = np.load(exp_path / 'renamed_metrics.npz')
			except FileNotFoundError:
				metrics = np.load(exp_path / 'metrics.npz')
			corrected_metrics_path = exp_path / 'corrected_metrics.npz'
			new_metrics = {
				'expected_reward': corrected_exp_reward,
				'expected_failure': metrics['expected_failure'],
				'std_reward': corrected_std_reward,
				'std_failure': metrics['std_failure']
			}
			np.savez(corrected_metrics_path, **new_metrics)