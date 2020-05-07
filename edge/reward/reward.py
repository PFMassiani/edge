from edge.utils import bind


class Reward:
    def get_reward(self, state, action, new_state, failed):
        raise NotImplementedError

    def __add__(self, other):
        def add_reward(instance, state, action, new_state, failed):
            return self.get_reward(state, action, new_state, failed) + \
                other.get_reward(state, action, new_state, failed)
        addition = Reward()
        bind(addition, add_reward, 'get_reward')

        return addition

    def __sub__(self, other):
        def sub_reward(instance, state, action, new_state, failed):
            return self.get_reward(state, action, new_state, failed) - \
                other.get_reward(state, action, new_state, failed)
        substraction = Reward()
        bind(substraction, sub_reward, 'get_reward')

        return substraction
