from numpy import array,ndarray
from numpy import float as npfloat

def ensure_np(x):
    if isinstance(x,ndarray):
        return x
    else:
        return array(x,dtype=npfloat)
