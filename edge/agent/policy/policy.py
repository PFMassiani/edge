class Policy:
    def __init__(self, stateaction_space):
        self.stateaction_space = stateaction_space

    def __call__(self, *args, **kwargs):
        return self.get_action(*args, **kwargs)

    def get_action(self, state):
        raise NotImplementedError

    def get_policy_map(self):
        raise NotImplementedError