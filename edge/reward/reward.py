from edge.utils import bind  # binds a method to an instance of a class


class Reward:
    """
    An abstract representation of the reward. Rewards can be added or substracted with the + and - operators.
    Subclasses should redefine the get_reward method.
    """
    def __init__(self, stateaction_space):
        self.stateaction_space = stateaction_space

    def get_reward(self, state, action, new_state, failed):
        """
        Abstract class. Returns the reward incurred at a given timestep. To satisfy the Markov property, the reward
        can only depend on the state, the action, the new state, and whether the agent has failed.
        :param state: the previous state
        :param action: the action taken
        :param new_state: the state we end up in
        :param failed: whether the agent has failed
        :return: the reward
        """
        raise NotImplementedError

    def __add__(self, other):
        """
        Overloads the `+` operator. The added rewards should have the same stateaction_space.
        :param other: other instance of Reward
        :return: the sum of the Rewards
        """
        if self.stateaction_space != other.stateaction_space:
            raise ValueError('Cannot add two rewards with different '
                             'state-action spaces')

        # get_reward is normally a method bound to the `self` object. Here, we need to dynamically define how
        # it behaves, so we use a local function, and bind it to the `addition` object.
        # Hence, add_reward needs to take `self` (here, `instance`) as first argument, since it ends up being a method
        def add_reward(instance, state, action, new_state, failed):
            return self.get_reward(state, action, new_state, failed) + \
                other.get_reward(state, action, new_state, failed)
        addition = Reward(self.stateaction_space)
        bind(addition, add_reward, 'get_reward')

        return addition

    def __sub__(self, other):
        """
        Overloads the `-` operator. The subbed rewards should have the same stateaction_space. The order of the
        substraction is: `self - other`
        :param other: other instance of Reward
        :return: the difference of the Rewards
        """
        if self.stateaction_space != other.stateaction_space:
            raise ValueError('Cannot add two rewards with different '
                             'state-action spaces')

        # Same comment as in __add__
        def sub_reward(instance, state, action, new_state, failed):
            return self.get_reward(state, action, new_state, failed) - \
                other.get_reward(state, action, new_state, failed)
        substraction = Reward(self.stateaction_space)
        bind(substraction, sub_reward, 'get_reward')

        return substraction
