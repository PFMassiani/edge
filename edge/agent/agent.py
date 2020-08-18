class Agent:
    """
    Base class for any agent. An Agent can be seen as a wrapper class, making three things work together:
        * the environment, in which it can step, and from which it collects reward and failure information
        * one or many policies, that it uses to decide what its next action is going to be
        * one or many models, that it uses to create an internal representation of its environment useful for the
        policies and that can be updated after every step
    In general, you should not put several Agents in one Environment object, because Environments have an internal
    state. To do that, you probably should define several Environments.
    """
    def __init__(self, env, *models):
        """
        Initializer
        :param env: the environment
        :param models: the list of models the agent has
        """
        self.env = env
        self.state = env.s
        self.last_action = None
        self.models = models

        self._training_mode = True

    def get_next_action(self):
        """ Abstract method
        Returns the next action taken by the Agent.
        :return: np.ndarray: the next action
        """
        raise NotImplementedError

    def update_models(self, state, action, new_state, reward, failed):
        """ Abstract method
        Update the models the Agent has
        :param state: the previous state
        :param action: the action taken
        :param new_state: the state the Agent ends up in
        :param reward: the collected reward
        :param failed: whether the Agent has failed
        """
        raise NotImplementedError

    def fit_models(self):
        """ Abstract method
        Fits the models to the problem. This method should typically be called before the Agent starts learning and
        updating its models. Its purpose is to fit the hyperparameters to the problem. In that case, the method will
        require additional data, coming from system knowledge or previous experiments.
        """
        raise NotImplementedError

    def reset(self, s=None):
        """
        Resets the Agent and the Environment
        :param s: (optional) the state in which to reset. If None, the default behaviour of the Environment is used.
        """
        self.state = self.env.reset(s)

    @property
    def failed(self):
        """
        Whether the Agent has failed
        :return: boolean
        """
        return self.env.has_failed

    @property
    def training_mode(self):
        return self._training_mode

    @training_mode.setter
    def training_mode(self, new_training_mode):
        self._training_mode = new_training_mode

    def step(self):
        """
        Chooses an action according to the policy, takes a step in the Environment, and updates the models. The action
        taken is available in self.last_action.
        :return: new_state, reward, failed
        """
        old_state = self.state
        action = self.get_next_action()
        new_state, reward, failed = self.env.step(action)
        if self.training_mode:
            self.update_models(old_state, action, new_state, reward, failed)
        self.state = new_state
        self.last_action = action
        return new_state, reward, failed
