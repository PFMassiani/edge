from edge import error


class Environment:
    def __init__(self, dynamics, reward, default_initial_state,
                 random_start=False):
        self.dynamics = dynamics
        self.reward = reward
        self.random_start = random_start
        if default_initial_state not in dynamics.stateaction_space.state_space:
            raise error.OutOfSpace('Default initial state is out of space')
        self._default_initial_state = default_initial_state
        self.reset()

    @property
    def stateaction_space(self):
        return self.dynamics.stateaction_space

    @property
    def state_space(self):
        return self.stateaction_space.state_space

    @property
    def action_space(self):
        return self.stateaction_space.action_space

    @property
    def has_failed(self):
        return (not self.feasible) or self.in_failure_state

    @property
    def in_failure_state(self):
        raise NotImplementedError

    @property
    def default_initial_state(self):
        return self._default_initial_state

    def reset(self, s=None):
        if s is not None and s not in self.stateaction_space.state_space:
            raise ValueError('Invalid state')
        elif s is not None:
            self.s = s
        elif self.random_start:
            self.s = self.stateaction_space.state_space.sample()
        else:
            self.s = self.default_initial_state
        self.feasible = self.dynamics.is_feasible_state(self.s)
        return self.s

    def step(self, action):
        old_state = self.s
        if not self.has_failed:
            self.s, self.feasible = self.dynamics.step(old_state, action)

        reward = self.reward.get_reward(old_state,
                                        action,
                                        self.s,
                                        self.has_failed
                                        )
        return self.s, reward, self.has_failed
