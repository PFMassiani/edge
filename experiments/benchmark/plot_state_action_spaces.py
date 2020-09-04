import matplotlib.pyplot as plt
from matplotlib import rc
from pathlib import Path
from edge.model.safety_models import SafetyTruth
from edge.graphics.subplotter import SafetyTruthSubplotter
from benchmark_environments import LowGoalHovership, LowGoalSlip
from benchmark_parameterizations import LOW_GOAL_HOVERSHIP_PARAMS, LOW_GOAL_SLIP_PARAMS
from edge.graphics.colors import corl_colors
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

VIBLY_DATA_PATH = Path('../../data/ground_truth/from_vibly')

hover_path = VIBLY_DATA_PATH / 'hover_map.pickle'
slip_path = VIBLY_DATA_PATH / 'slip_map.pickle'

output_path = Path('.') / 'state_action_spaces'
output_path.mkdir(exist_ok=True)

for envname, envconstr, param, tpath in [('hovership', LowGoalHovership, LOW_GOAL_HOVERSHIP_PARAMS, hover_path),
										 ('slip', LowGoalSlip, LOW_GOAL_SLIP_PARAMS, slip_path)]:
	env = envconstr(**param)
	truth = SafetyTruth(env)
	truth.from_vibly_file(tpath)
	subplotter = SafetyTruthSubplotter(truth, corl_colors)

	figure = plt.figure(constrained_layout=True, figsize=(5.5, 4.8))
	# gs = figure.add_gridspec(1, 2, width_ratios=[3, 1])

	ax_Q = figure.add_subplot()

	subplotter.draw_on_axs(ax_Q, None)
	ax_Q.tick_params(direction='in', top=True, right=True)
	# ax_S.tick_params(direction='in', left=False)
	ax_Q.set_xlabel(r'action space $A$')
	ax_Q.set_ylabel(r'state space $S$')
	# ax_S.set_xlabel(r'$\Lambda$')

	output_file = output_path / (envname + '.pdf')
	figure.savefig(str(output_file), format='pdf')