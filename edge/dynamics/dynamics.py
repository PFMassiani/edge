import numpy as np

from edge.space import StateActionSpace
from .event import event, event_based


class DiscreteTimeDynamics:
    def __init__(self, stateaction_space):
        self.stateaction_space = stateaction_space

    def step(self, state, action):
        raise NotImplementedError


@event_based
class TimestepIntegratedDynamics(DiscreteTimeDynamics):
    def __init__(self, stateaction_space, integration_time):
        self.integration_time = integration_time
        super(TimestepIntegratedDynamics, self).__init__(stateaction_space)
    # TODO
