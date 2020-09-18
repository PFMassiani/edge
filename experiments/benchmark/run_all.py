import os

def run_sim_batch(envname, aname):
	print(f'================= {envname}:{aname} =================')
	for i in range(1):
		print(f'SIMULATION #{i}')
		command = f'./run_experiment.sh {i} benchmark benchmark_main.py {envname} {aname}'
		os.system(command)

configs = [
	# ('low_goal_hovership', 'q_learner'),
	# ('penalized_hovership', 'q_learner'),
	('low_goal_hovership', 'safety_values_switcher'),
	('low_goal_hovership', 'safety_q_learner'),
	('low_goal_hovership', 'soft_hard_learner'),
]

for envname, aname in configs:
	run_sim_batch(envname, aname)