class Policy:
    def __init__(self, env):
        self.env = env

    def __call__(self, *args, **kwargs):
        return self.get_action(*args, **kwargs)

    def get_action(self, state):
        raise NotImplementedError

    def get_policy_map(self):
        raise NotImplementedError