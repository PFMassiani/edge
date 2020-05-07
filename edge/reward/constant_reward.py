from .reward import Reward


class ConstantReward(Reward):
    def __init__(self, constant, rewarded_set=None, unrewarded_set=None):
        if rewarded_set is not None and unrewarded_set is not None:
            raise ValueError('`rewarded_set` and `unrewarded_set` cannot be'
                             'both specified')
        super(ConstantReward, self).__init__()
        self.constant = constant
        self.rewarded = rewarded_set
        self.unrewarded = unrewarded_set

    def get_reward(self, state, action, new_state, failed):
        if self.rewarded is not None:
            reward = new_state in self.rewarded.state_space
        elif self.unrewarded is not None:
            reward = new_state not in self.unrewarded.state_space
        else:
            reward = True

        if reward:
            return self.constant
        else:
            return 0
