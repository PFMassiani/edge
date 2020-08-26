from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from benchmark_parameterizations import ENVIRONMENTS
from benchmark_simulation import BenchmarkSingleSimulation as b


base_path = Path(__file__).absolute().parent


def ls(p):
	return p.iterdir()

def get_name(p):
	return str(p).split('/')[-1]

def get_metrics_dict_from_list(metrics_list):
	n_exps = len(metrics_list)
	keys = list(metrics_list[0].keys())
	metrics = {k:
		[metrics_list[rid][k] for rid in range(n_exps)]
		for k in keys
	}
	return metrics

def get_metrics_of_experiment(envname, aname, rid):
	exp_path = base_path/envname/aname/rid/'metrics.npz'
	metrics = np.load(exp_path.absolute())
	return metrics

def get_rids(envname, aname):
	apath = base_path/envname/aname
	return ls(apath)

def get_run_agents(envname):
	envpath = base_path/envname
	dirs = [p for p in ls(envpath) if p.is_dir()]
	envs = [p for p in dirs if get_name(p) != 'metrics']
	return envs

def get_run_envs():
	dirs = [p for p in ls(base_path) if p.is_dir()]
	envs = [p for p in dirs if get_name(p) in ENVIRONMENTS]
	return envs


if __name__=='__main__':
	n_sigma = 2
	metrics = {}
	for envpath in get_run_envs():
		envname = get_name(envpath)
		x = {}
		metrics[envname] = {}
		for apath in get_run_agents(envname):
			aname = get_name(apath)
			x[aname] = {}
			metrics_list = [get_metrics_of_experiment(envname, aname, rid)
							for rid in get_rids(envname, aname)]
			metrics[envname][aname] = get_metrics_dict_from_list(metrics_list)
			for k in metrics[envname][aname].keys():
				x[aname][k] = np.array(metrics[envname][aname][k][0])[:, 0]
				metrics[envname][aname][k] = np.array(
					metrics[envname][aname][k]
				)[:, :, 1]


		for exp_name, std_name in [(b.EXP_REWARD_MNAME, b.STD_REWARD_MNAME),
									(b.EXP_FAILURE_MNAME, b.STD_FAILURE_MNAME)]:
			save_dir = base_path/envname/'metrics'
			save_dir.mkdir(exist_ok=True)
			savepath = save_dir / exp_name

			agents = list(metrics[envname].keys())
			exp = {aname: metrics[envname][aname][exp_name].mean(axis=0).squeeze() for aname in agents}
			std = {aname: metrics[envname][aname][std_name].mean(axis=0).squeeze() for aname in agents}
			
			figure = plt.figure(constrained_layout=True, figsize=(4, 4))

			ax = figure.add_subplot()
			ax.tick_params(direction='in')
			ax.grid(True)

			for aname in agents:
				lower = exp[aname] - n_sigma * std[aname]
				upper = exp[aname] + n_sigma * std[aname]
				print(lower.shape, upper.shape)
				ax.plot(x[aname][exp_name], exp[aname], label=aname)
				#ax.fill_between(x[aname][std_name], lower, upper, alpha=0.5)
			ax.legend(loc='best')

			figure.savefig(str(savepath), format='pdf')