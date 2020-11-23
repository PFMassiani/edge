from matplotlib import rc
import matplotlib.pyplot as plt
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
rc('pdf', **{'fonttype': 42})
rc('ps', **{'fonttype': 42})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('pdf', fonttype=42)
plt.rc('ps', fonttype=42)

from .plotter import Plotter
from .safety_plotter import SafetyPlotter, DetailedSafetyPlotter
from .q_value_plotter import QValuePlotter, DiscreteQValuePlotter
from .duality_plotter import DualityPlotter
from .q_value_and_safety import QValueAndSafetyPlotter, SoftHardPlotter
from .sample_plotter import SamplePlotter
from .reward_failure_plotter import RewardFailurePlotter