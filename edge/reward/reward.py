from edge.utils import bind


class Reward:
    def __init__(self, stateaction_space):
        self.stateaction_space = stateaction_space

    def get_reward(self, state, action, new_state, failed):
        raise NotImplementedError

    def __add__(self, other):
        if self.stateaction_space != other.stateaction_space:
            raise ValueError('Cannot add two rewards with different '
                             'state-action spaces')

        def add_reward(instance, state, action, new_state, failed):
            return self.get_reward(state, action, new_state, failed) + \
                other.get_reward(state, action, new_state, failed)
        addition = Reward(self.stateaction_space)
        bind(addition, add_reward, 'get_reward')

        return addition

    def __sub__(self, other):
        if self.stateaction_space != other.stateaction_space:
            raise ValueError('Cannot add two rewards with different '
                             'state-action spaces')

        def sub_reward(instance, state, action, new_state, failed):
            return self.get_reward(state, action, new_state, failed) - \
                other.get_reward(state, action, new_state, failed)
        substraction = Reward(self.stateaction_space)
        bind(substraction, sub_reward, 'get_reward')

        return substraction
