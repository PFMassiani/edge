class Agent:
    def __init__(self, env):
        self.env = env
        self.state = env.s

    def get_next_action(self):
        raise NotImplementedError

    def update_models(self, state, action, new_state, reward, failed):
        raise NotImplementedError

    def fit_models(self):
        raise NotImplementedError

    def reset(self):
        self.state = self.env.reset()

    @property
    def failed(self):
        return self.env.has_failed

    def step(self):
        action = self.get_next_action()
        new_state, reward, failed = self.env.step(action)
        self.update_models(self.state, action, new_state, reward, failed)
        self.state = new_state
        return new_state, reward, failed
