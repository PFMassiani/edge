from edge import error


class Environment:
    def __init__(self, dynamics, reward, default_initial_state,
                 random_start=False):
        self.dynamics = dynamics
        self.reward = reward
        self.random_start = random_start
        if default_initial_state not in dynamics.stateaction_space.state_space:
            raise error.OutOfSpace('Default initial state is out of space')
        self.default_initial_state = default_initial_state
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
    def state_index(self):
        return self.state_space.get_index_of(self.s)

    def reset(self, state_index=None):
        if state_index is not None:
            self.s = self.state_space[state_index]
        elif self.random_start:
            self.s = self.stateaction_space.state_space.sample()
        else:
            self.s = self.default_initial_state
        self.feasible = self.dynamics.is_feasible_state(self.s)
        return self.state_index

    def step(self, action_index):
        old_state = self.s
        action = self.action_space[action_index]
        if not self.has_failed:
            self.s, self.feasible = self.dynamics.step(old_state, action)

        reward = self.reward.get_reward(old_state,
                                        action,
                                        self.s,
                                        self.has_failed
                                        )
        return self.s, reward, self.has_failed
