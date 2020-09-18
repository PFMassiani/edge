from pathlib import Path
import numpy as np
from benchmark_simulation import BenchmarkSingleSimulation as b


def get_name(p):
	return str(p.absolute()).split('/')[-1]

base_path = Path('./results/')

for envpath in base_path.iterdir():
	for apath in envpath.iterdir():
		aname = get_name(apath)
		if aname == 'metrics':
			continue
		for expname in apath.iterdir():
			for fpath in expname.iterdir():
				fname = get_name(fpath)
				if '.npz' not in fname:
					continue
				metrics = np.load(fpath)
				metrics_names_correction = {
					b.EXP_REWARD_MNAME: b.EXP_REWARD_MNAME,
					b.STD_REWARD_MNAME: b.STD_REWARD_MNAME,
					b.EXP_FAILURE_MNAME: b.EXP_FAILURE_MNAME,
					b.STD_FAILURE_MNAME: b.STD_FAILURE_MNAME,
					b.Q_V_Q_C_MNAME: b.Q_C_Q_V_MNAME,
					b.Q_C_Q_V_MNAME: b.Q_V_Q_C_MNAME,
				}
				corr_metrics = {
					metrics_names_correction[mname]: metrics[metrics_names_correction[mname]]
					for mname in metrics.keys()
				}
				corr_metrics_path = expname / 'renamed_metrics.npz'
				np.savez(corr_metrics_path, **corr_metrics)