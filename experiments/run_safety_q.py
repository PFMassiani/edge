import os

for i in range(10):
	print(f'SIMULATION #{i}')
	command = f'./run_experiment.sh {i} benchmark benchmark_main.py low_goal_hovership safety_q_learner'
	os.system(command)