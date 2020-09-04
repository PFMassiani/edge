from pathlib import Path
import numpy as np

def ls(p):
	return p.iterdir()

def get_name(p):
	return str(p.absolute()).split('/')[-1]


if __name__ == '__main__':
	base_path = Path('./results/')
	n_fails = {}
	for envpath in ls(base_path):
		n_fails[get_name(envpath)] = {}
		for apath in ls(envpath):
			dirname = get_name(apath)
			if dirname == 'metrics':
				continue # Go to next iteration
			n_fails_agent = []
			for exp_path in ls(apath):
				print(get_name(exp_path))
				trainpath = exp_path / 'samples' / 'training.npz'
				training_samples = np.load(str(trainpath))
				n_fails_agent.append(sum(
					int(training_samples[f'failed_EPISODE_{t}'][-1])
					for t in range(500)
				)/500)
			n_fails[get_name(envpath)][get_name(apath)] = np.mean(n_fails_agent)

	for env, envfails in n_fails.items():
		print(f'----- {env}:')
		for agent, afails in envfails.items():
			print(f'{agent}: {100*afails:.2f} %')