from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from benchmark_parameterizations import ENVIRONMENTS, \
	Q_LEARNER, SAFETY_Q_LEARNER, SOFT_HARD_LEARNER, SAFETY_VALUES_SWITCHER,\
	LOW_GOAL_SLIP, PENALIZED_SLIP, LOW_GOAL_HOVERSHIP, PENALIZED_HOVERSHIP

from benchmark_simulation import BenchmarkSingleSimulation as b
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

METRICS_META = {
	b.EXP_REWARD_MNAME: {'name':r'Expected reward', 'symbol':r'$G$', 'fname':'expected_reward.pdf'},
    b.EXP_FAILURE_MNAME: {'name':r'Expected failure', 'symbol':r'$F$', 'fname':'expected_failure.pdf'},
    b.STD_REWARD_MNAME: {'name':r'Reward standard deviation', 'symbol':r'$\sigma_G$', 'fname':'std_reward.pdf'},
    b.STD_FAILURE_MNAME: {'name':r'Failure standard deviation', 'symbol':r'$\sigma_F$', 'fname':'std_failure.pdf'},
    b.Q_C_Q_V_MNAME: {'name':r'Conservativeness', 'symbol':r'$K$', 'fname':'conservativeness.pdf'},
    b.Q_V_Q_C_MNAME: {'name':r'Negligence', 'symbol':r'$N$', 'fname':'negligence.pdf'},
}

AGENTS_META = {
	Q_LEARNER: {'name': r'Q-Learning'},
	SAFETY_Q_LEARNER: {'name': r'Safety Q-Learning'},
	SOFT_HARD_LEARNER: {'name': r'Soft Safety Q-Learning'},
	SAFETY_VALUES_SWITCHER: {'name': r'Safety-Q switch'},
}

base_path = Path(__file__).absolute().parent / 'results'


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
	if envname.startswith('penalized_'):
		exp_path = base_path/envname/aname/rid/'corrected_metrics.npz'
	else:
		exp_path = base_path/envname/aname/rid
		has_renamed = False
		for f in exp_path.iterdir():
			fname = get_name(f)
			if fname == 'corrected_metrics.npz':
				has_renamed = True
				break
		if has_renamed:
			exp_path = exp_path/'renamed_metrics.npz'
		else:
			exp_path = exp_path/'metrics.npz'
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

def get_label(aname):
	if Q_LEARNER in aname:
		splitted = aname.split('_p_')
		if len(splitted) == 2: # Penalized Q-Learning
			aname, p = splitted
			label = AGENTS_META[aname]['name'] + r' ($p=' + p + r'$)'
		else:
			label = AGENTS_META[aname]['name']
	else:
		label = AGENTS_META[aname]['name']
	return label

def is_penalized(aname):
	return len(aname.split('_p_')) == 2


def export_legend(legend, filename="legend.png", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, format="pdf", bbox_inches=bbox)
    legend.remove()


if __name__=='__main__':
	n_sigma = 2
	metrics = {}
	x = {}
	for envpath in get_run_envs():
		envname = get_name(envpath)
		metrics[envname] = {}
		x[envname] = {}
		for apath in get_run_agents(envname):
			aname = get_name(apath)
			x[envname][aname] = {}
			metrics_list = [get_metrics_of_experiment(envname, aname, rid)
							for rid in get_rids(envname, aname)]
			metrics[envname][aname] = get_metrics_dict_from_list(metrics_list)
			for k in metrics[envname][aname].keys():
				x[envname][aname][k] = np.array(metrics[envname][aname][k][0])[:, 0]
				metrics[envname][aname][k] = np.array(
					metrics[envname][aname][k]
				)[:, :, 1]

	metrics[LOW_GOAL_HOVERSHIP].update(metrics.pop(PENALIZED_HOVERSHIP))
	metrics[LOW_GOAL_SLIP].update(metrics.pop(PENALIZED_SLIP))
	x[LOW_GOAL_HOVERSHIP].update(x.pop(PENALIZED_HOVERSHIP))
	x[LOW_GOAL_SLIP].update(x.pop(PENALIZED_SLIP))
	for envname in metrics.keys():
		envprefix = envname.split('_')[-1] + '_'
		for exp_name, std_name in [(b.EXP_REWARD_MNAME, b.STD_REWARD_MNAME),
									(b.EXP_FAILURE_MNAME, b.STD_FAILURE_MNAME)]:
			save_dir = base_path/envname/'metrics'
			save_dir.mkdir(exist_ok=True)

			agents = list(metrics[envname].keys())
			exp = {aname: metrics[envname][aname][exp_name].mean(axis=0).squeeze() for aname in agents}
			std = {aname: metrics[envname][aname][std_name].mean(axis=0).squeeze() for aname in agents}
			
			def plot_metric(met, mname, save_legend=True):
				figure = plt.figure(figsize=(4, 4))

				ax = figure.add_subplot()
				ax.tick_params(direction='in')
				ax.grid(True)

				for aname in agents:
					if True:#not is_penalized(aname) or exp_name == b.EXP_FAILURE_MNAME:
						ax.plot(x[envname][aname][mname], met[aname], label=get_label(aname))
						ax.set_xlabel(r'Episodes')
						ax.set_ylabel(METRICS_META[mname]['symbol'])
						# ax.fill_between(x[envname][aname][std_name], lower, upper, alpha=0.5)
				ax.set_title(METRICS_META[mname]['name'])
				savepath = save_dir / (envprefix + METRICS_META[mname]['fname'])
				if save_legend:
					legend = figure.legend(loc=7)
					figure.tight_layout()
					figure.subplots_adjust(right=0.45)
					legend_savepath = save_dir / ('legend_' + envprefix[:-1] + '.pdf')
					export_legend(legend, str(legend_savepath))
					figure.subplots_adjust(right=0.95)
				else:
					figure.tight_layout()
					figure.subplots_adjust(right=0.95)
				figure.savefig(str(savepath), format='pdf')

			plot_metric(exp, exp_name, True)
			plot_metric(std, std_name, False)

		def plot_metric(mname_to_plot):
			figure = plt.figure(figsize=(4, 4))
			#gs = figure.add_gridspec(1, 2)
			ax = figure.add_subplot()
			figure_has_plots = False
			for aname in metrics[envname].keys():
				metrics_names = list(metrics[envname][aname].keys())
				if mname_to_plot in metrics_names:
					q_metric = metrics[envname][aname][mname_to_plot].mean(axis=0).squeeze()
					ax.plot(x[envname][aname][mname_to_plot], q_metric, label=get_label(aname))
					figure_has_plots = True
				else:
					pass
			if figure_has_plots:
				ax.grid(True)
				ax.set_xlabel(r'Episodes')
				ax.set_ylabel(METRICS_META[mname_to_plot]['symbol'])
				# ax.legend(bbox_to_anchor=(1.04, 1))
				ax.set_title(METRICS_META[mname_to_plot]['name'])
				# figure.legend(loc=7)
				figure.tight_layout()
				# figure.subplots_adjust(right=0.65)
				savepath = save_dir / (envprefix + METRICS_META[mname_to_plot]['fname'])
				figure.savefig(str(savepath), format='pdf')

		plot_metric(b.Q_C_Q_V_MNAME)
		plot_metric(b.Q_V_Q_C_MNAME)
