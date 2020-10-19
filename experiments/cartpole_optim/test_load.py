import numpy as np
from pathlib import Path

from lander_model import SymmetricMaternCosGPSARSA, SymmetricMaternCosSafety
from lander_env import PenalizedLunarLander

env = PenalizedLunarLander(10000, (10, 10, 10, 10, 10, 10, 10, 10), 5)
x_seed = np.array([
    [0,   0, 0, 0, 0, 0, 0],
    [0, 1.4, 0, 0, 0, 0, 0]
])
y_seed = np.array([200, 100])

p = Path('./lander_1601653695/models')
q_model = SymmetricMaternCosGPSARSA.load(p/'Q_model', env, x_seed, y_seed)

print(q_model)

x_seed = np.array([
    [0, 1.4, 0, 0, 0, 0, 2]
])
y_seed = np.array([1])
safety_model = SymmetricMaternCosSafety.load(p/'safety_model', env, 0.8, x_seed, y_seed)

print(safety_model)