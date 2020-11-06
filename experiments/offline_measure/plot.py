from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from edge.dataset import Dataset

basepath = Path('/home/pifou/Documents/Max_Planck/WD/edge/experiments/offline_measure/offline_1604494254')
figpath = basepath / 'figs'

PRIORS = True
SAFE_RESET = True
save_suffix = '_safe_reset' if SAFE_RESET else ''
priors_fname = f'priors_evaluations{save_suffix}.csv'
learned_fname = f'learned_models_evaluations{save_suffix}.csv'
fname = priors_fname if PRIORS else learned_fname
ds = Dataset.load(basepath/'data'/fname)
df = ds.df

gr = df.groupby('Model number')

fig_r, ax_r = plt.subplots(figsize=(8,6))
fig_f, ax_f = plt.subplots(figsize=(8,6))


def plot_and_save(suffix, save_suffix):
	# Get groups with highest end value
	reward = 'reward' + suffix
	failed = 'failed' + suffix
	dfs = {label: df for label, df in gr}
	sorted_label_gr = sorted(
		dfs.items(),
		key=lambda x: x[1].loc[x[1].index[-1], reward],
		reverse=True
	)

	i = 0
	TOP = 5
	for label, df in sorted_label_gr:
		detailed = (i < TOP or int(label) == -1)
		alpha = 1 if detailed else 0.1
		lab = label if detailed else '_nolegend_'
		df.plot(x='episode', y=reward, ax=ax_r, label=lab, alpha=alpha)
		df.plot(x='episode', y=failed, ax=ax_f, label=lab, alpha=alpha)
		i += 1
	ax_r.set_title('Average ' + reward)
	ax_f.set_title('Average ' + failed)
	ax_r.grid()
	ax_f.grid()
	plt.legend()

	prefix = 'priors_' if PRIORS else ''
	savename = prefix + '{}' + save_suffix + '.pdf'
	fig_r.savefig(str(figpath/savename.format('reward')), format='pdf')
	fig_f.savefig(str(figpath/savename.format('failure')), format='pdf')
	# plt.show()


plot_and_save('', '')
plot_and_save(suffix=' (safe reset)', save_suffix='_safe_reset')
