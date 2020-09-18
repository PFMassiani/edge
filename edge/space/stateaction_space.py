from .space import ProductSpace


class StateActionSpace(ProductSpace):
    """Wrapper around a ProductSpace to give it the state-action space terminology.
    This class only provides helper methods and accessors"""
    def __init__(self, state_space, action_space):
        super(StateActionSpace, self).__init__(
            state_space,
            action_space
        )
        self.state_space = state_space
        self.action_space = action_space

    def __getitem__(self, index):
        if len(index) == 2:
            index = self.get_stateaction(*index)
        return super(StateActionSpace, self).__getitem__(index)

    @staticmethod
    def from_product(product_space):
        """
        Creates a StateActionSpace from a product space. The product space should be a product of 2 spaces
        :param product_space: the product space
        :return: StateActionSpace
        """
        n = len(product_space.sets)
        if not len(product_space.sets) == 2:
            raise IndexError(f'Expected a product of 2 spaces, got {n}')
        return StateActionSpace(*product_space.sets)

    def get_action(self, stateaction):
        """
        Extracts the action from a stateaction
        :param stateaction:
        :return: the action
        """
        return self.get_component(stateaction, self.action_space)

    def get_state(self, stateaction):
        """
        Extracts the state from a stateaction
        :param stateaction:
        :return: the state
        """
        return self.get_component(stateaction, self.state_space)

    def get_stateaction(self, state, action):
        """
        Creates a StateActionSpace element from a state and an action
        :param state: the state
        :param action: the action
        :return: the stateaction
        """
        return self.from_components(state, action)

    def get_tuple(self, stateaction):
        """
        Splits a StateActionSpace element into a state and an action
        :param stateaction: the stateaction
        :return: tuple: state, action
        """
        return (self.get_state(stateaction), self.get_action(stateaction))

    def get_state_index(self, stateaction_index):
        """
        Extracts the index of the state in its original space from a stateaction index
        :param stateaction_index: the index of the stateaction
        :return: the index of the state in the state space
        """
        return self.get_index_component(stateaction_index, self.state_space)

    def get_action_index(self, stateaction_index):
        """
        Extracts the index of the action in its original space from a stateaction index
        :param stateaction_index: the index of the stateaction
        :return: the index of the action in the state space
        """
        return self.get_index_component(stateaction_index, self.action_space)

    def get_index_tuple(self, stateaction_index):
        """
        Extracts the indexes of the state and the action in their original spaces from a stateaction index
        :param stateaction_index: the index of the stateaction
        :return: tuple: state index, action index
        """
        return (self.get_state_index(stateaction_index),
                self.get_action_index(stateaction_index))
