from .space import ProductSpace


class StateActionSpace(ProductSpace):
    def __init__(self, state_space, action_space):
        super(StateActionSpace, self).__init__(
            state_space,
            action_space
        )
        self.state_space = state_space
        self.action_space = action_space

    @staticmethod
    def from_product(product_space):
        n = len(product_space.sets)
        if not len(product_space.sets) == 2:
            raise IndexError(f'Expected a product of 2 spaces, got {n}')
        return StateActionSpace(*product_space.sets)

    def get_action(self, stateaction):
        return self.get_component(stateaction, self.action_space)

    def get_state(self, stateaction):
        return self.get_component(stateaction, self.state_space)

    def get_stateaction(self, state, action):
        return self.from_components(state, action)

    def get_tuple(self, stateaction):
        return (self.get_state(stateaction), self.get_action(stateaction))
