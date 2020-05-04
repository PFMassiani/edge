from itertools import product
import numpy as np

class Space:
    def __init__(self):
        pass
    def contains(self,x):
        raise NotImplementedError
    def __contains__(self,x):
        return self.contains(elem)
    def sample(self):
        raise NotImplementedError

class DiscreteSpace(Space):
    def __init__(self):
        super(DiscreteSpace,self).__init__()
    def __getitem__(self,index):
        raise NotImplementedError
    def indexof(self,x):
        raise NotImplementedError
    def get_index_iterator(self):
        raise NotImplementedError

class DiscreteProductSpace(DiscreteSpace):
    def __init__(self,*sets):
        super(DiscreteProductSpace,self).__init__()
        self.sets = sets

    def contains(self,*coordinates):
        if len(coordinates) != len(self.sets):
            raise ValueError(f'Size mismatch: expected {len(self.sets)} coordinates, received {len(coordinates)}')
        coordinates_and_sets = zip(coordinates,self.sets)
        isin = lambda x,set : x in set
        return all(map(isin,coordinates_and_sets))

    def sample(self):
        sample_set = lambda s:s.sample()
        return tuple(map(sample_set,self.sets))

    def __getitem__(self,index):
        if len(index) != len(self.sets):
            raise ValueError(f'Size mismatch: expected {len(self.sets)} coordinates, received {len(index)}')
        index_and_sets = zip(index,self.sets)
        get_item = lambda ind,s:s[ind]
        return list(map(get_item,index_and_sets))

    def indexof(self,x):
        if len(x) != len(self.sets):
            raise ValueError(f'Size mismatch: expected {len(self.sets)} coordinates, received {len(coordinates)}')
        x_and_sets = zip(x,self.sets)
        indexof_coord = lambda x,s: s.indexof(x)
        return list(map(indexof_coord,x))

    def get_index_iterator(self):
        get_idx_iter = lambda s:s.get_index_iterator()
        return product(tuple(map(get_idx_iter,self.sets)))

class DiscreteStateActionSpace(DiscreteProductSpace):
    def __init__(self,state_space,action_space):
        super(DiscreteStateActionSpace,self).__init__(state_space,action_space)
        self.state_space = state_space
        self.action_space = action_space

class DiscreteSubspace(DiscreteSpace):
    def __init__(self,space,idx_in_subspace):
        super(DiscreteSubspace,self).__init__()
        # TODO
