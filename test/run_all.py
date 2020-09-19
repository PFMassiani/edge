from pathlib import Path
import os

p = Path('.')

ignored_tests = [
	'imports_test',
	'mlp_test',
	'mujoco_test',
]
long_tests = [
	#'hyperparameters_test',
	#'simulation_test',
	'ground_truth_test',
]
ignored_tests = ignored_tests + long_tests

def get_name(p):
	return str(p).split('/')[-1]

command = "python -m unittest {}"
for fpath in p.iterdir():
	fname = get_name(fpath)
	if not fname.endswith('py'):
		continue
	elif fname == 'run_all.py':
		continue

	testname = fname[:-3]  # Without the .py extension
	if testname in ignored_tests:
		continue

	exit_code = os.system(command.format(fname))
	if exit_code != 0:
		print(f'ERROR: {testname} failed')
