import logging
logger = logging.getLogger(__name__)

from .policy import Policy, RandomPolicy
from .greedy import ConstrainedEpsilonGreedy, EpsilonGreedy
from .safety import SafetyMaximization, SafetyActiveSampling
from .gaussian_policy import GaussianPolicy
from .bayesian import ExpectedImprovementPolicy, SafetyInformationMaximization
from .dlqr import DLQRPolicy
try:
    from .multilayer_perceptron import MLPPolicy, MultilayerPerceptron
except ImportError:
    logger.warning("Could not import multi-layer perceptrons: "
                   "safety-starter-agents is probably missing.")