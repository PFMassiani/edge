from edge import error


class Environment:
    """ Puts together a Dynamics and a Reward objects
    Relevant properties are:
    :param s: the current state of the environment
    :param has_failed: whether self.s is a failure state
    """
    def __init__(self, dynamics, reward, default_initial_state,
                 random_start=False, reward_done_threshold=None,
                 steps_done_threshold=None):
        """ Initializer
        :param dynamics: the Dynamics object the environment wraps
        :param reward: the Reward object the environment wraps
        :param default_initial_state: the default initial state (only used if random_start = False)
        :param random_start: whether to initalize the agent randomly
        :param reward_done_threshold: at what reward threshold the environment
            is considered done.
        :param steps_done_threshold: the maximal number of steps
        """
        self.dynamics = dynamics
        self.reward = reward
        self.random_start = random_start
        if default_initial_state not in dynamics.stateaction_space.state_space:
            raise error.OutOfSpace('Default initial state is out of space')
        self.default_initial_state = default_initial_state
        self.reward_done_threshold = reward_done_threshold
        self.reward_accumulator = 0
        self.steps_done_threshold = steps_done_threshold
        self.n_steps = 0
        self.reset()

    @property
    def stateaction_space(self):
        """ Accessor to the stateaction_space
        :return: stateaction_space
        """
        return self.dynamics.stateaction_space

    @property
    def state_space(self):
        """ Accessor to the state_space
        :return: state_space
        """
        return self.stateaction_space.state_space

    @property
    def action_space(self):
        """ Accessor to the action_space
        :return: action_space
        """
        return self.stateaction_space.action_space

    @property
    def has_failed(self):
        """ Whether the state is feasible and the environment has not failed. In general, the state is always
        feasible, so this is equivalent to self.in_failure_state. However, calling has_failed should be preferred
        since it enables additional checks on the current state.
        :return: boolean: has_failed
        """
        return (not self.feasible) or self.in_failure_state

    def is_failure_state(self, state):
        """ Whether any state is a failure state. Agents should not use this in general.
        :param state: the state to check
        :return: boolean
        """
        raise NotImplementedError

    @property
    def in_failure_state(self):
        """ Whether the current state is a failure state.
        :return: boolean
        """
        return self.is_failure_state(self.s)

    @property
    def done(self):
        """ Whether the environment is done and `reset` should be called.
        This is more general than failing: an environment might be done
        whereas the agent has not failed, for example because the reward
        limit is exceeded.
        By default, an environment is done iff it has failed or has exceeded the
        reward thresholde (if it is specified). """
        if self.reward_done_threshold is not None:
            reward_done = self.reward_accumulator >= self.reward_done_threshold
        else:
            reward_done = False
        if self.steps_done_threshold is not None:
            steps_done = self.steps_done_threshold >= self.n_steps
        else:
            steps_done = False
        return reward_done or steps_done or self.in_failure_state

    @property
    def state_index(self):
        """ The index of the current state
        :return: int or tuple
        """
        return self.state_space.get_index_of(self.s)

    def reset(self, s=None):
        """ Resets the internal state of the environment
        If a parameter is specified, the state is reset to that value. If not, and self.random_start = True, then the
        internal state is randomly sampled in the state space. Otherwise, the internal state takes the default value.
        :param s: optional: the state where to initialize
        :return: the internal state of the environment (after reinitialization)
        """
        if s is not None:
            self.s = s
        elif self.random_start:
            self.s = self.stateaction_space.state_space.sample()
        else:
            self.s = self.default_initial_state
        self.feasible = self.dynamics.is_feasible_state(self.s)
        self.reward_accumulator = 0
        self.n_steps = 0
        return self.s

    def step(self, action):
        """ Take a step.
        To redefine this function, remember that the actual computations for the dynamics and the reward should be
        done in the corresponding object.
        :param action: The action taken
        :return: s: the new state
        :return: reward: the reward sampled
        :return: has_failed: whether the environment has failed
        """
        old_state = self.s
        if not self.has_failed:
            self.s, self.feasible = self.dynamics.step(old_state, action)

        reward = self.reward.get_reward(old_state,
                                        action,
                                        self.s,
                                        self.has_failed
                                        )
        self.reward_accumulator += reward
        self.n_steps += 1
        return self.s, reward, self.has_failed

    def render(self):
        pass

    def compute_dynamics_map(self):
        return self.dynamics.compute_map()

    def linearization(self):
        """
        Returns the linearization matrices of the environment.
        If the linearized dynamics are given by dx/dt = Ax + Bu, this returns
        matrices A and B.
        :return: the linearized dynamics
        """
        raise NotImplementedError