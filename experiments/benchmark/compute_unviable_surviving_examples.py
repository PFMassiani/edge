from pathlib import Path
import numpy as np

def ls(p):
	return p.iterdir()

def get_name(p):
	return str(p.absolute()).split('/')[-1]

STATES = 'states'

if __name__ == '__main__':
	base_path = Path('./results/penalized_hovership/q_learner_p_10000')
	n_surviving = []
	for exp_path in ls(base_path):
		if get_name(exp_path) == 'n_surviving.npy':
			continue
		samples_path = exp_path / 'samples'
		n_surviving_exp = []
		for measurement_episode_file in ls(samples_path):
			measurement_name = get_name(measurement_episode_file)[:-4]  # Remove npz extension
			if measurement_name == 'training':
				continue
			measurement_number = int(measurement_name.split('_')[-1])
			meas = np.load(measurement_episode_file)
			n_episodes = 20
			measurements = [{STATES: []} for ep in range(n_episodes)]
			for key in meas.keys():
					mname, ep = key.split('_EPISODE_')
					if mname == STATES:
						ep = int(ep)
						measurements[ep][mname] = meas[key]
			n_surviving_exp.append(len([1 for meas_ep in measurements if (meas_ep[STATES][-1] < 0.75) and (len(meas_ep[STATES]) == 10)]))
		n_surviving.append(n_surviving_exp)
	n_surviving = np.array(n_surviving).T
	savepath = base_path / 'n_surviving.npy'
	np.save(str(savepath), n_surviving)
