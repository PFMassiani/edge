from .reward import Reward


class ConstantReward(Reward):
    """
    Defines a constant reward on the StateActionSpace or a subset of it.
    Subsets can be defined by explicitly specifying a Space object, or by providing a function that checks whether
    the step should be rewarded.
    """
    def __init__(self, stateaction_space, constant, rewarded_set=None,
                 unrewarded_set=None, reward_condition=None):
        """
        Initializer. If `rewarded_set`, `unrewarded_set` and `reward_condition` are all None, the whole Space is
        rewarded.
        :param stateaction_space: the stateaction_space
        :param constant: the constant the reward is equal to
        :param rewarded_set: StateActionSpace (optional): the stateactions that should incur a reward
        :param unrewarded_set: StateActionSpace (optional): the stateactions that should not incur a reward. Its
            complementary is rewarded
        :param reward_condition: function(state, action, new_state, failed) -> boolean. Whether a step should be
            rewarded
        """

        if rewarded_set is not None and unrewarded_set is not None:
            raise ValueError('`rewarded_set` and `unrewarded_set` cannot be'
                             'both specified')
        super(ConstantReward, self).__init__(stateaction_space)
        self.constant = constant
        self.rewarded = rewarded_set
        self.unrewarded = unrewarded_set
        self.reward_condition = reward_condition

    def get_reward(self, state, action, new_state, failed):
        if self.rewarded is not None:
            reward = new_state in self.rewarded.state_space
        elif self.unrewarded is not None:
            reward = new_state not in self.unrewarded.state_space
        elif self.reward_condition is not None:
            reward = self.reward_condition(state, action, new_state, failed)
        else:
            reward = True

        if reward:
            return self.constant
        else:
            return 0
