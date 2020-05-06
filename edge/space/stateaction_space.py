from .space import ProductSpace


class StateActionSpace(ProductSpace):
    def __init__(self, state_space, action_space):
        super(DiscreteStateActionSpace, self).__init__(
            state_space,
            action_space
        )
        self.state_space = state_space
        self.action_space = action_space

    def action(self, stateaction):
        return self.get_projection_on_space(stateaction, self.action_space)

    def state(self, stateaction):
        return self.get_projection_on_space(stateaction, self.state_space)
