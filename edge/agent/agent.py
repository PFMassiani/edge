class Agent:
    def __init__(self, env):
        self.env = env
        self.state = env.s
        self.last_action = None

    def get_next_action(self):
        raise NotImplementedError

    def update_models(self, state, action, new_state, reward, failed):
        raise NotImplementedError

    def fit_models(self):
        raise NotImplementedError

    def reset(self, s=None):
        self.state = self.env.reset(s)

    @property
    def failed(self):
        return self.env.has_failed

    def step(self):
        old_state = self.state
        action = self.get_next_action()
        new_state, reward, failed = self.env.step(action)
        self.update_models(old_state, action, new_state, reward, failed)
        self.state = new_state
        self.last_action = action
        return new_state, reward, failed
